"""FastAPI boundary for server-to-server RAG answers."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline.rag_answer import answer_with_rag


logger = logging.getLogger(__name__)
INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise RuntimeError("RAG_INTERNAL_API_KEY is required to start the RAG API service.")

SourceType = Literal["pdf", "note", "annotation_comment"]


class RagAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1)
    source_types: list[SourceType] | None = None
    top_k: int = Field(default=8, ge=1, le=20)
    pdf_ids: list[str] | None = None


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
        return answer_with_rag(
            query=request.query.strip(),
            user_id=request.user_id,
            source_types=request.source_types,
            top_k=request.top_k,
            pdf_ids=request.pdf_ids,
        )
    except Exception:
        logger.exception("RAG answer generation failed")
        raise HTTPException(
            status_code=500,
            detail="RAG answer generation failed.",
        ) from None
