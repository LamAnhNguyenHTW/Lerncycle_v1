from __future__ import annotations

import json

from rag_pipeline.document_primer import (
    DocumentPrimer,
    build_document_primer,
    document_primer_content_hash,
)
from rag_pipeline.models import RagChunk, SourceRef


def _chunk(
    content: str,
    *,
    heading_path: list[str] | None = None,
    page_index: int | None = None,
    content_hash: str | None = None,
) -> RagChunk:
    return RagChunk(
        source=SourceRef(
            user_id="user-1",
            source_type="pdf",
            source_id="pdf-1",
            pdf_id="pdf-1",
        ),
        content=content,
        content_hash=content_hash or str(abs(hash(content))),
        page_index=page_index,
        heading_path=heading_path or [],
        metadata={"filename": "Process Mining.pdf"},
    )


class FakeLlm:
    def __init__(self, response: dict | str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return json.dumps(self.response) if isinstance(self.response, dict) else self.response


def test_fallback_primer_uses_headings_terms_and_representative_chunks() -> None:
    full_text = "Process Mining analyses event logs and discovers process models. " * 40
    primer = build_document_primer(
        [
            _chunk(full_text, heading_path=["Grundlagen", "Event Logs"], page_index=0),
            _chunk("Alpha Miner Petri nets conformance checking", heading_path=["Methoden"], page_index=4),
        ],
        llm_client=None,
    )

    assert primer.title == "Process Mining.pdf"
    assert "Grundlagen" in primer.main_topics
    assert "Event Logs" in primer.main_topics
    assert "process" in [term.lower() for term in primer.key_terms]
    assert primer.page_ranges == [{"start": 1, "end": 5}]
    assert full_text not in primer.summary
    assert len(primer.summary) < 700


def test_llm_primer_is_validated_and_content_hash_is_set() -> None:
    llm = FakeLlm(
        {
            "title": "Event Logs",
            "summary": "Das Dokument erklärt Event Logs und deren Nutzung.",
            "main_topics": ["Event Logs"],
            "key_terms": ["Case ID"],
            "learning_objectives": ["Event Logs erklären"],
            "page_ranges": [{"start": 1, "end": 2}],
        }
    )

    primer = build_document_primer(
        [_chunk("Event Logs have case ids.", heading_path=["Event Logs"], page_index=0)],
        llm_client=llm,
    )

    assert primer == DocumentPrimer(
        title="Event Logs",
        summary="Das Dokument erklärt Event Logs und deren Nutzung.",
        main_topics=["Event Logs"],
        key_terms=["Case ID"],
        learning_objectives=["Event Logs erklären"],
        page_ranges=[{"start": 1, "end": 2}],
        content_hash=document_primer_content_hash(
            [_chunk("Event Logs have case ids.", heading_path=["Event Logs"], page_index=0)]
        ),
    )
    assert llm.calls


def test_llm_failure_falls_back_to_deterministic_primer() -> None:
    primer = build_document_primer(
        [_chunk("BPMN gateways events tasks", heading_path=["BPMN"], page_index=1)],
        llm_client=FakeLlm("not json"),
    )

    assert primer.title == "Process Mining.pdf"
    assert "BPMN" in primer.main_topics
    assert primer.content_hash
