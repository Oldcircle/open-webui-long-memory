import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence


class LLMComplete(Protocol):
    async def __call__(self, messages: list[dict[str, Any]]) -> str: ...


class Embedding(Protocol):
    async def __call__(
        self, texts: str | list[str], *, prefix: Optional[str] = None, user: Any = None
    ) -> Any: ...


class VectorStore(Protocol):
    def upsert(self, collection_name: str, items: list[dict[str, Any]]): ...

    def search(self, collection_name: str, vectors: list[list[float]], limit: int): ...


@dataclass(frozen=True)
class LongMemoryConfig:
    enabled: bool = True
    k: int = 3
    collection_name_prefix: str = "user-long-memory-"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    max_queries: int = 5
    max_memories_per_turn: int = 6


@dataclass(frozen=True)
class NeedCheckResult:
    need: bool
    queries: list[str]


@dataclass(frozen=True)
class RecallItem:
    id: str
    text: str
    metadata: dict[str, Any]
    score: Optional[float] = None


@dataclass(frozen=True)
class RecallResult:
    items: list[RecallItem]
    queries: list[str]


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        it = _normalize_query(it)
        if not it:
            continue
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _flatten_vector_search_results(results: Any) -> list[RecallItem]:
    items: list[RecallItem] = []

    if results is None:
        return items

    documents = getattr(results, "documents", None)
    metadatas = getattr(results, "metadatas", None)
    ids = getattr(results, "ids", None)
    distances = getattr(results, "distances", None)

    if isinstance(documents, list) and documents:
        docs0 = documents[0] if isinstance(documents[0], list) else documents
        metas0 = []
        if isinstance(metadatas, list) and metadatas:
            metas0 = metadatas[0] if isinstance(metadatas[0], list) else metadatas
        ids0 = []
        if isinstance(ids, list) and ids:
            ids0 = ids[0] if isinstance(ids[0], list) else ids
        dists0 = []
        if isinstance(distances, list) and distances:
            dists0 = distances[0] if isinstance(distances[0], list) else distances

        for idx, doc in enumerate(docs0):
            text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            meta = metas0[idx] if idx < len(metas0) and isinstance(metas0[idx], dict) else {}
            _id = str(ids0[idx]) if idx < len(ids0) else str(uuid.uuid4())
            score = None
            if idx < len(dists0) and isinstance(dists0[idx], (int, float)):
                score = float(dists0[idx])
            items.append(RecallItem(id=_id, text=text, metadata=meta, score=score))
        return items

    if isinstance(results, dict) and "documents" in results:
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        ids = results.get("ids") or []
        dists = results.get("distances") or []
        docs0 = docs[0] if docs and isinstance(docs[0], list) else docs
        metas0 = metas[0] if metas and isinstance(metas[0], list) else metas
        ids0 = ids[0] if ids and isinstance(ids[0], list) else ids
        dists0 = dists[0] if dists and isinstance(dists[0], list) else dists
        for idx, doc in enumerate(docs0):
            text = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            meta = metas0[idx] if idx < len(metas0) and isinstance(metas0[idx], dict) else {}
            _id = str(ids0[idx]) if idx < len(ids0) else str(uuid.uuid4())
            score = None
            if idx < len(dists0) and isinstance(dists0[idx], (int, float)):
                score = float(dists0[idx])
            items.append(RecallItem(id=_id, text=text, metadata=meta, score=score))
        return items

    return items


class NomicSentenceTransformerEmbedding:
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self._model = None
        self._lock = asyncio.Lock()

    async def _get_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            return self._model

    async def __call__(
        self, texts: str | list[str], *, prefix: Optional[str] = None, user: Any = None
    ) -> Any:
        model = await self._get_model()
        single = isinstance(texts, str)
        input_texts = [texts] if single else list(texts)

        def encode():
            kwargs = {"batch_size": self.batch_size}
            if prefix:
                kwargs["prompt"] = prefix
            return model.encode(input_texts, **kwargs).tolist()

        vectors = await asyncio.to_thread(encode)
        return vectors[0] if single else vectors


class LongMemoryService:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding: Embedding,
        config: LongMemoryConfig = LongMemoryConfig(),
    ):
        self.vector_store = vector_store
        self.embedding = embedding
        self.config = config

    def _collection_name(self, user_id: str) -> str:
        return f"{self.config.collection_name_prefix}{user_id}"

    async def check_need(self, llm: LLMComplete, messages: list[dict[str, Any]]) -> NeedCheckResult:
        recent = messages[-10:] if len(messages) > 10 else messages

        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个长期记忆路由器。给定对话上下文，判断是否需要从长期记忆中召回信息。"
                    "只输出JSON，格式：{\"need\": boolean, \"queries\": [string]}。"
                    "queries用于向量检索，必须是简短分点，不要重复，最多5条；如果need为false，queries必须为空数组。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"messages": recent}, ensure_ascii=False),
            },
        ]

        text = await llm(prompt)
        obj = _extract_first_json_object(text) or {}
        need = bool(obj.get("need", False))
        queries = obj.get("queries", [])
        if not isinstance(queries, list):
            queries = []
        queries = _dedupe_keep_order([str(x) for x in queries])
        if not need:
            queries = []
        queries = queries[: self.config.max_queries]
        return NeedCheckResult(need=need, queries=queries)

    async def recall(
        self,
        *,
        user_id: str,
        queries: list[str],
        k: int,
        user: Any = None,
    ) -> RecallResult:
        queries = _dedupe_keep_order(queries)[: self.config.max_queries]
        if not queries:
            return RecallResult(items=[], queries=[])

        vectors = await self.embedding(queries, user=user)
        if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
            vectors = [vectors]

        all_items: list[RecallItem] = []
        seen_texts: set[str] = set()
        collection = self._collection_name(user_id)
        k = max(1, int(k))
        per_query_limit = max(k, k * 3)

        for q_idx, vec in enumerate(vectors[: len(queries)]):
            try:
                results = self.vector_store.search(
                    collection_name=collection,
                    vectors=[vec],
                    limit=per_query_limit,
                )
            except Exception:
                continue

            picked = 0
            for item in _flatten_vector_search_results(results):
                if picked >= k:
                    break
                key = item.text.strip()
                if not key:
                    continue
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                all_items.append(item)
                picked += 1

        return RecallResult(items=all_items, queries=queries)

    async def summarize(self, llm: LLMComplete, *, user_text: str, assistant_text: str) -> list[str]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个长期记忆提取器。基于一轮对话（用户输入+助手输出），提取未来对话中稳定且有用的信息。"
                    "要求：尽量简短，每条不超过30字；避免重复；不要记录临时信息或纯闲聊；不要输出解释。"
                    "只输出JSON，格式：{\"memories\": [string]}。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"user": user_text or "", "assistant": assistant_text or ""},
                    ensure_ascii=False,
                ),
            },
        ]

        text = await llm(prompt)
        obj = _extract_first_json_object(text) or {}
        memories = obj.get("memories", [])
        if not isinstance(memories, list):
            memories = []
        memories = _dedupe_keep_order([str(x) for x in memories])
        return memories[: self.config.max_memories_per_turn]

    async def store(
        self,
        *,
        user_id: str,
        memories: list[str],
        user: Any = None,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> list[str]:
        memories = _dedupe_keep_order(memories)[: self.config.max_memories_per_turn]
        if not memories:
            return []

        vectors = await self.embedding(memories, user=user)
        if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
            vectors = [vectors]

        created_at = int(time.time())
        items = []
        ids = []
        for idx, mem in enumerate(memories):
            mem_id = str(uuid.uuid4())
            ids.append(mem_id)
            items.append(
                {
                    "id": mem_id,
                    "text": mem,
                    "vector": vectors[idx],
                    "metadata": {
                        "created_at": created_at,
                        **({"chat_id": chat_id} if chat_id else {}),
                        **({"message_id": message_id} if message_id else {}),
                    },
                }
            )

        self.vector_store.upsert(
            collection_name=self._collection_name(user_id),
            items=items,
        )
        return ids
