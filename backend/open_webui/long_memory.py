import asyncio
import ast
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from open_webui.env import LONG_MEMORY_DEBUG
from open_webui.long_memory_prompts import (
    CHECK_NEED_SYSTEM_PROMPT,
    MERGE_MEMORIES_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    WRITE_CHECK_AND_EXTRACT_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)


class LLMComplete(Protocol):
    async def __call__(self, messages: list[dict[str, Any]]) -> str: ...


class Embedding(Protocol):
    async def __call__(
        self, texts: str | list[str], *, prefix: Optional[str] = None, user: Any = None
    ) -> Any: ...


class VectorStore(Protocol):
    def upsert(self, collection_name: str, items: list[dict[str, Any]]): ...

    def search(self, collection_name: str, vectors: list[list[float]], limit: int): ...


class LongMemorySQLStore(Protocol):
    def insert_new_long_memory(
        self,
        user_id: str,
        content: str,
        *,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Any: ...


@dataclass(frozen=True)
class LongMemoryConfig:
    enabled: bool = True
    k: int = 3
    collection_name_prefix: str = "user-long-memory-"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    max_queries: int = 5
    max_memories_per_turn: int = 30


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

@dataclass(frozen=True)
class WriteCheckResult:
    need: bool
    memories: list[str]


def _parse_json_like_object(candidate: str) -> Optional[dict[str, Any]]:
    candidate = (candidate or "").strip()
    if not candidate:
        return None

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    try:
        normalized = re.sub(r"\btrue\b", "True", candidate, flags=re.I)
        normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.I)
        normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.I)
        obj = ast.literal_eval(normalized)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    text = text or ""
    if not text:
        return None

    for match in re.finditer(r"\{[\s\S]*?\}", text):
        obj = _parse_json_like_object(match.group(0))
        if obj is not None:
            return obj

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return _parse_json_like_object(text[start : end + 1])


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
        sql_store: Optional[LongMemorySQLStore] = None,
        config: LongMemoryConfig = LongMemoryConfig(),
    ):
        self.vector_store = vector_store
        self.embedding = embedding
        self.sql_store = sql_store
        self.config = config

    def _collection_name(self, user_id: str) -> str:
        return f"{self.config.collection_name_prefix}{user_id}"

    async def check_need(self, llm: LLMComplete, messages: list[dict[str, Any]]) -> NeedCheckResult:
        user_text = ""
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                user_text = content
            else:
                user_text = json.dumps(content, ensure_ascii=False)
            break
        return await self.check_need_by_user_text(llm, user_text=user_text)

    async def check_need_by_user_text(self, llm: LLMComplete, *, user_text: str) -> NeedCheckResult:
        prompt = [
            {
                "role": "system",
                "content": CHECK_NEED_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps({"user": user_text or ""}, ensure_ascii=False),
            },
        ]

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory check_need prompt: user_len=%s user_text=%s",
                len(user_text or ""),
                (user_text or "")[:500],
            )

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
        if LONG_MEMORY_DEBUG:
            log.info("long_memory check_need parsed: need=%s queries=%s", need, queries)
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

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory recall start: user_id=%s collection=%s k=%s queries=%s",
                user_id,
                self._collection_name(user_id),
                k,
                queries,
            )

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
            except Exception as e:
                if LONG_MEMORY_DEBUG:
                    log.info(
                        "long_memory recall search failed: query_idx=%s query=%s err=%s",
                        q_idx,
                        queries[q_idx] if q_idx < len(queries) else "",
                        repr(e),
                    )
                continue

            picked = 0
            flattened = _flatten_vector_search_results(results)
            if LONG_MEMORY_DEBUG:
                log.info(
                    "long_memory recall search result: query_idx=%s query=%s raw_items=%s",
                    q_idx,
                    queries[q_idx] if q_idx < len(queries) else "",
                    len(flattened),
                )
                log.info(
                    "long_memory recall top_items: %s",
                    [
                        {
                            "id": it.id,
                            "score": it.score,
                            "text": (it.text or "")[:200],
                            "metadata": it.metadata,
                        }
                        for it in flattened[: min(10, len(flattened))]
                    ],
                )
            for item in flattened:
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

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory recall picked: items=%s queries=%s",
                len(all_items),
                queries,
            )
        return RecallResult(items=all_items, queries=queries)

    async def find_similar_existing_memories(
        self,
        *,
        user_id: str,
        memories: list[str],
        k_per_memory: int = 5,
        min_similarity: float = 0.8,
        user: Any = None,
    ) -> list[RecallItem]:
        memories = _dedupe_keep_order(memories)
        if not memories:
            return []

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory find_similar start: user_id=%s collection=%s memories=%s k_per=%s min_score=%s",
                user_id,
                self._collection_name(user_id),
                [(m or "")[:200] for m in memories],
                k_per_memory,
                min_similarity,
            )

        k_per_memory = max(1, int(k_per_memory))
        min_similarity = float(min_similarity)

        vectors = await self.embedding(memories, user=user)
        if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
            vectors = [vectors]

        collection = self._collection_name(user_id)
        out: list[RecallItem] = []
        seen_ids: set[str] = set()

        for vec in vectors[: len(memories)]:
            try:
                results = self.vector_store.search(
                    collection_name=collection,
                    vectors=[vec],
                    limit=k_per_memory,
                )
            except Exception as e:
                if LONG_MEMORY_DEBUG:
                    log.info("long_memory find_similar search failed: err=%s", repr(e))
                continue

            picked = 0
            for item in _flatten_vector_search_results(results):
                if picked >= k_per_memory:
                    break
                score = item.score
                if score is None or float(score) < min_similarity:
                    continue
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                out.append(item)
                picked += 1

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory find_similar picked: items=%s details=%s",
                len(out),
                [
                    {
                        "id": it.id,
                        "score": it.score,
                        "text": (it.text or "")[:200],
                        "metadata": it.metadata,
                    }
                    for it in out[: min(10, len(out))]
                ],
            )
        return out

    async def merge_memories(
        self,
        llm: LLMComplete,
        *,
        new_memories: list[str],
        old_memories: list[str],
    ) -> list[str]:
        new_memories = _dedupe_keep_order([str(x) for x in (new_memories or [])])
        old_memories = _dedupe_keep_order([str(x) for x in (old_memories or [])])
        if not new_memories and not old_memories:
            return []

        prompt = [
            {
                "role": "system",
                "content": MERGE_MEMORIES_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"new_memories": new_memories, "old_memories": old_memories},
                    ensure_ascii=False,
                ),
            },
        ]

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory merge prompt: new=%s old=%s",
                [(m or "")[:200] for m in new_memories],
                [(m or "")[:200] for m in old_memories],
            )

        text = await llm(prompt)
        obj = _extract_first_json_object(text) or {}
        memories = obj.get("memories", [])
        if not isinstance(memories, list):
            memories = []
        memories = _dedupe_keep_order([str(x) for x in memories])
        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory merge parsed: merged=%s",
                [(m or "")[:200] for m in memories],
            )
        return memories[: self.config.max_memories_per_turn]

    async def check_and_summarize(
        self, llm: LLMComplete, *, user_text: str, assistant_text: str
    ) -> WriteCheckResult:
        prompt = [
            {
                "role": "system",
                "content": WRITE_CHECK_AND_EXTRACT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps({"user": user_text or ""}, ensure_ascii=False),
            },
        ]

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory write_check prompt: user_len=%s user_text=%s",
                len(user_text or ""),
                (user_text or "")[:500],
            )

        text = await llm(prompt)
        obj = _extract_first_json_object(text) or {}
        need = bool(obj.get("need", False))
        memories = obj.get("memories", [])
        if not isinstance(memories, list):
            memories = []
        memories = _dedupe_keep_order([str(x) for x in memories])
        if not need:
            memories = []
        memories = memories[: self.config.max_memories_per_turn]
        if LONG_MEMORY_DEBUG:
            log.info("long_memory write_check parsed: need=%s memories=%s", need, memories)
        return WriteCheckResult(need=need, memories=memories)

    async def summarize(self, llm: LLMComplete, *, user_text: str, assistant_text: str) -> list[str]:
        prompt = [
            {
                "role": "system",
                "content": SUMMARIZE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps({"user": user_text or ""}, ensure_ascii=False),
            },
        ]

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory summarize prompt: user_len=%s user_text=%s",
                len(user_text or ""),
                (user_text or "")[:500],
            )

        text = await llm(prompt)
        obj = _extract_first_json_object(text) or {}
        memories = obj.get("memories", [])
        if not isinstance(memories, list):
            memories = []
        memories = _dedupe_keep_order([str(x) for x in memories])
        if LONG_MEMORY_DEBUG:
            log.info("long_memory summarize parsed: memories=%s", memories)
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

        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory store start: user_id=%s memories=%s chat_id=%s message_id=%s",
                user_id,
                [(m or "")[:200] for m in memories],
                chat_id,
                message_id,
            )

        now = int(time.time())
        ids: list[str] = []
        created_ats: list[int] = []
        updated_ats: list[int] = []

        for mem in memories:
            mem_id = str(uuid.uuid4())
            created_at = now
            updated_at = now

            if self.sql_store is not None:
                try:
                    rec = self.sql_store.insert_new_long_memory(
                        user_id,
                        mem,
                        chat_id=chat_id,
                        message_id=message_id,
                    )
                    if rec is not None:
                        mem_id = str(getattr(rec, "id", mem_id))
                        created_at = int(getattr(rec, "created_at", created_at))
                        updated_at = int(getattr(rec, "updated_at", updated_at))
                except Exception:
                    pass

            ids.append(mem_id)
            created_ats.append(created_at)
            updated_ats.append(updated_at)

        vectors = await self.embedding(memories, user=user)
        if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
            vectors = [vectors]

        items = []
        for idx, mem in enumerate(memories):
            items.append(
                {
                    "id": ids[idx],
                    "text": mem,
                    "vector": vectors[idx],
                    "metadata": {
                        "created_at": created_ats[idx],
                        "updated_at": updated_ats[idx],
                        **({"chat_id": chat_id} if chat_id else {}),
                        **({"message_id": message_id} if message_id else {}),
                    },
                }
            )

        self.vector_store.upsert(
            collection_name=self._collection_name(user_id),
            items=items,
        )
        if LONG_MEMORY_DEBUG:
            log.info(
                "long_memory store done: ids=%s count=%s collection=%s",
                ids,
                len(ids),
                self._collection_name(user_id),
            )
        return ids
