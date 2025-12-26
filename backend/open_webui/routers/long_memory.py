import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from open_webui.long_memory import (
    LongMemoryConfig,
    LongMemoryService,
    NomicSentenceTransformerEmbedding,
)
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
        config=LongMemoryConfig(),
    )
    request.app.state.LONG_MEMORY_SERVICE = service
    return service


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

    result = await service.check_need(llm, form_data.messages)
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

    need = await service.check_need(llm, form_data.messages)
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

    user_text = _get_last_role_text(form_data.messages, "user")
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
