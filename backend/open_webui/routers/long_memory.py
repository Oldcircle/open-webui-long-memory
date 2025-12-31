import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from open_webui.long_memory import (
    LongMemoryConfig,
    LongMemoryService,
    NomicSentenceTransformerEmbedding,
)
from open_webui.models.memories import LongMemories, LongMemoryModel
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.utils.auth import get_verified_user
from open_webui.utils.chat import generate_chat_completion


log = logging.getLogger(__name__)

router = APIRouter()


def _get_long_memory_service(request: Request) -> LongMemoryService:
    service = getattr(request.app.state, "LONG_MEMORY_SERVICE", None)
    if service is not None:
        return service

    embedding = NomicSentenceTransformerEmbedding()
    service = LongMemoryService(
        vector_store=VECTOR_DB_CLIENT,
        embedding=embedding,
        sql_store=LongMemories,
        config=LongMemoryConfig(),
    )
    request.app.state.LONG_MEMORY_SERVICE = service
    return service


def _collection_name(service: LongMemoryService, user_id: str) -> str:
    prefix = getattr(getattr(service, "config", None), "collection_name_prefix", "user-long-memory-")
    return f"{prefix}{user_id}"


async def _extract_assistant_content_from_completion_response(response) -> Optional[str]:
    if isinstance(response, dict):
        if response.get("choices") and response["choices"][0].get("message", {}).get(
            "content"
        ):
            return response["choices"][0]["message"]["content"]
        return None

    if isinstance(response, JSONResponse) and isinstance(response.body, bytes):
        try:
            data = json.loads(response.body.decode("utf-8", "replace"))
            if data.get("choices") and data["choices"][0].get("message", {}).get(
                "content"
            ):
                return data["choices"][0]["message"]["content"]
        except Exception:
            return None

    return None


@router.get("/", response_model=list[LongMemoryModel])
async def get_long_memories(user=Depends(get_verified_user)):
    return LongMemories.get_long_memories_by_user_id(user.id)


class AddLongMemoryForm(BaseModel):
    content: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None


class LongMemoryUpdateModel(BaseModel):
    content: Optional[str] = None


@router.post("/add", response_model=Optional[LongMemoryModel])
async def add_long_memory(
    request: Request,
    form_data: AddLongMemoryForm,
    user=Depends(get_verified_user),
):
    service = _get_long_memory_service(request)
    memory = LongMemories.insert_new_long_memory(
        user.id,
        form_data.content,
        chat_id=form_data.chat_id,
        message_id=form_data.message_id,
    )

    if memory is None:
        return None

    vector = await service.embedding(memory.content, user=user)

    VECTOR_DB_CLIENT.upsert(
        collection_name=_collection_name(service, user.id),
        items=[
            {
                "id": memory.id,
                "text": memory.content,
                "vector": vector,
                "metadata": {
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    **({"chat_id": memory.chat_id} if memory.chat_id else {}),
                    **({"message_id": memory.message_id} if memory.message_id else {}),
                },
            }
        ],
    )

    return memory


@router.post("/{memory_id}/update", response_model=Optional[LongMemoryModel])
async def update_long_memory_by_id(
    memory_id: str,
    request: Request,
    form_data: LongMemoryUpdateModel,
    user=Depends(get_verified_user),
):
    if form_data.content is None:
        raise HTTPException(status_code=400, detail="No updates provided")

    service = _get_long_memory_service(request)
    memory = LongMemories.update_long_memory_by_id_and_user_id(
        memory_id, user.id, form_data.content
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    vector = await service.embedding(memory.content, user=user)

    VECTOR_DB_CLIENT.upsert(
        collection_name=_collection_name(service, user.id),
        items=[
            {
                "id": memory.id,
                "text": memory.content,
                "vector": vector,
                "metadata": {
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    **({"chat_id": memory.chat_id} if memory.chat_id else {}),
                    **({"message_id": memory.message_id} if memory.message_id else {}),
                },
            }
        ],
    )

    return memory


@router.post("/reset", response_model=bool)
async def reset_long_memory_from_vector_db(request: Request, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)
    VECTOR_DB_CLIENT.delete_collection(_collection_name(service, user.id))

    memories = LongMemories.get_long_memories_by_user_id(user.id)
    if not memories:
        return True

    vectors = await service.embedding([m.content for m in memories], user=user)
    if not isinstance(vectors, list) or (vectors and not isinstance(vectors[0], list)):
        vectors = [vectors]

    VECTOR_DB_CLIENT.upsert(
        collection_name=_collection_name(service, user.id),
        items=[
            {
                "id": memory.id,
                "text": memory.content,
                "vector": vectors[idx],
                "metadata": {
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    **({"chat_id": memory.chat_id} if memory.chat_id else {}),
                    **({"message_id": memory.message_id} if memory.message_id else {}),
                },
            }
            for idx, memory in enumerate(memories)
        ],
    )

    return True


@router.delete("/delete/user", response_model=bool)
async def delete_long_memories_by_user_id(request: Request, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)
    result = LongMemories.delete_long_memories_by_user_id(user.id)

    if result:
        try:
            VECTOR_DB_CLIENT.delete_collection(_collection_name(service, user.id))
        except Exception as e:
            log.error(e)
        return True

    return False


@router.delete("/{memory_id}", response_model=bool)
async def delete_long_memory_by_id(
    request: Request, memory_id: str, user=Depends(get_verified_user)
):
    service = _get_long_memory_service(request)
    result = LongMemories.delete_long_memory_by_id_and_user_id(memory_id, user.id)

    if result:
        VECTOR_DB_CLIENT.delete(
            collection_name=_collection_name(service, user.id), ids=[memory_id]
        )
        return True

    return False


class CheckNeedForm(BaseModel):
    model: str
    messages: list[dict[str, Any]]


@router.post("/check")
async def check_need(request: Request, form_data: CheckNeedForm, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)

    async def llm(messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": form_data.model,
            "messages": messages,
            "stream": False,
            "metadata": {"task": "long_memory_check"},
        }
        res = await generate_chat_completion(
            request=request, form_data=payload, user=user, bypass_filter=True
        )
        content = await _extract_assistant_content_from_completion_response(res)
        return content or ""

    user_text = ""
    for msg in reversed(form_data.messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        user_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        break

    result = await service.check_need_by_user_text(llm, user_text=user_text)
    return {"need": result.need, "queries": result.queries}


class SummarizeForm(BaseModel):
    model: str
    user_text: str
    assistant_text: str


@router.post("/summarize")
async def summarize(
    request: Request, form_data: SummarizeForm, user=Depends(get_verified_user)
):
    service = _get_long_memory_service(request)

    async def llm(messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": form_data.model,
            "messages": messages,
            "stream": False,
            "metadata": {"task": "long_memory_summarize"},
        }
        res = await generate_chat_completion(
            request=request, form_data=payload, user=user, bypass_filter=True
        )
        content = await _extract_assistant_content_from_completion_response(res)
        return content or ""

    memories = await service.summarize(
        llm,
        user_text=form_data.user_text,
        assistant_text=form_data.assistant_text,
    )
    return {"memories": memories}


class StoreForm(BaseModel):
    memories: list[str]
    chat_id: Optional[str] = None
    message_id: Optional[str] = None


@router.post("/store")
async def store(request: Request, form_data: StoreForm, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)
    ids = await service.store(
        user_id=user.id,
        memories=form_data.memories,
        user=user,
        chat_id=form_data.chat_id,
        message_id=form_data.message_id,
    )
    return {"ids": ids}


class RecallForm(BaseModel):
    queries: list[str]
    k: Optional[int] = 3


@router.post("/recall")
async def recall(request: Request, form_data: RecallForm, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)
    result = await service.recall(
        user_id=user.id,
        queries=form_data.queries,
        k=int(form_data.k or 3),
        user=user,
    )
    return {
        "queries": result.queries,
        "items": [{"id": it.id, "text": it.text, "metadata": it.metadata} for it in result.items],
    }


def _get_last_role_text(messages: list[dict[str, Any]], role: str) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != role:
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    return ""


class DemoForm(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    k: Optional[int] = 3


@router.post("/demo")
async def demo(request: Request, form_data: DemoForm, user=Depends(get_verified_user)):
    service = _get_long_memory_service(request)

    async def llm(messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": form_data.model,
            "messages": messages,
            "stream": False,
            "metadata": {"task": "long_memory_demo"},
        }
        res = await generate_chat_completion(
            request=request, form_data=payload, user=user, bypass_filter=True
        )
        content = await _extract_assistant_content_from_completion_response(res)
        return content or ""

    user_text = _get_last_role_text(form_data.messages, "user")
    need = await service.check_need_by_user_text(llm, user_text=user_text)
    recalled = (
        await service.recall(
            user_id=user.id,
            queries=need.queries,
            k=int(form_data.k or 3),
            user=user,
        )
        if need.need and need.queries
        else None
    )

    assistant_text = _get_last_role_text(form_data.messages, "assistant")
    memories = (
        await service.summarize(llm, user_text=user_text, assistant_text=assistant_text)
        if user_text and assistant_text
        else []
    )

    return {
        "need": need.need,
        "queries": need.queries,
        "recalled": (
            {
                "queries": recalled.queries,
                "items": [
                    {"id": it.id, "text": it.text, "metadata": it.metadata}
                    for it in recalled.items
                ],
            }
            if recalled
            else {"queries": [], "items": []}
        ),
        "summarized_memories": memories,
    }
