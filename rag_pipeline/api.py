"""FastAPI boundary for server-to-server RAG answers."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from rag_pipeline.config import WorkerConfig
from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.reranker import create_reranker


logger = logging.getLogger(__name__)
INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise RuntimeError("RAG_INTERNAL_API_KEY is required to start the RAG API service.")

SourceType = Literal["pdf", "note", "annotation_comment"]


class RecentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class RagAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1)
    source_types: list[SourceType] | None = None
    top_k: int = Field(default=8, ge=1, le=20)
    pdf_ids: list[str] | None = None
    recent_messages: list[RecentMessage] = Field(default_factory=list, max_length=10)
    reranking_enabled: bool | None = None
    reranking_candidate_k: int | None = Field(default=None, ge=1, le=50)
    reranking_top_k: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def validate_reranking_bounds(self) -> "RagAnswerRequest":
        if self.reranking_enabled is True:
            candidate_k = self.reranking_candidate_k or 30
            top_k = self.reranking_top_k or 8
            if candidate_k < top_k:
                raise ValueError("reranking_candidate_k must be >= reranking_top_k")
        return self


class RagAnswerResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


app = FastAPI(title="LearnCycle RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


def require_internal_auth(authorization: str | None = Header(default=None)) -> None:
    """Validate the internal bearer token used by Next.js."""
    expected = f"Bearer {INTERNAL_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/rag/answer",
    response_model=RagAnswerResponse,
    dependencies=[Depends(require_internal_auth)],
)
def rag_answer(request: RagAnswerRequest) -> dict[str, Any]:
    try:
        config = WorkerConfig.from_env()
        reranking_enabled = (
            request.reranking_enabled
            if request.reranking_enabled is not None
            else config.reranking_enabled
        )
        reranking_candidate_k = request.reranking_candidate_k or config.reranking_candidate_k
        reranking_top_k = request.reranking_top_k or config.reranking_top_k
        if reranking_enabled and reranking_candidate_k < reranking_top_k:
            raise HTTPException(
                status_code=422,
                detail="reranking_candidate_k must be >= reranking_top_k",
            )
        reranker = create_reranker(
            provider=config.reranking_provider,
            model=config.reranking_model,
            enabled=reranking_enabled,
        )
        return answer_with_rag(
            query=request.query.strip(),
            user_id=request.user_id,
            source_types=request.source_types,
            top_k=request.top_k,
            pdf_ids=request.pdf_ids,
            recent_messages=[message.model_dump() for message in request.recent_messages],
            reranker=reranker,
            reranking_enabled=reranking_enabled,
            reranking_candidate_k=reranking_candidate_k,
            reranking_top_k=reranking_top_k,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="RAG answer generation failed.",
        ) from None
