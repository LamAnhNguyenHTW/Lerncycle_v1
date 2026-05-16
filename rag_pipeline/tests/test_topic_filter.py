from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import ExtractedConcept, ExtractedLearningObjective, ExtractedTopic
from rag_pipeline.learning_structure.topic_filter import (
    filter_candidates,
    is_generic_summary,
    is_low_content_topic,
    is_slide_artifact,
)


@dataclass(frozen=True)
class FilterMeta:
    document_page_count: int | None = 100
    learning_graph_min_topic_chars: int = 200


def _summary(title: str = "Process Discovery") -> str:
    return f"{title} explains how learners can interpret process data and connect evidence to analytical decisions."


def _topic(**overrides) -> ExtractedTopic:
    title = overrides.get("title", "Process Discovery")
    data = {
        "topic_id": make_extracted_topic_id("source", "group", sum(ord(char) for char in str(title))),
        "title": title,
        "summary": _summary(str(title)),
        "level": 1,
        "parent_title": None,
        "chunk_ids": ["c1"],
        "page_start": 20,
        "page_end": 21,
        "confidence": 0.8,
        "group_id": "group",
        "heading_path": [str(title)],
        "order_hint": 1,
    }
    data.update(overrides)
    if "page_start" in overrides and "page_end" not in overrides:
        data["page_end"] = overrides["page_start"]
    return ExtractedTopic(**data)


def _concept(topic_title: str = "Process Discovery") -> ExtractedConcept:
    return ExtractedConcept(
        name="Event Log",
        definition="Recorded process events",
        explanation="The evidence used for analysis",
        topic_title=topic_title,
        chunk_ids=["c1"],
        difficulty="medium",
        confidence=0.8,
    )


def _objective(topic_title: str = "Process Discovery") -> ExtractedLearningObjective:
    return ExtractedLearningObjective(
        objective="Explain process discovery from event data",
        topic_title=topic_title,
        bloom_level="understand",
        chunk_ids=["c1"],
        confidence=0.8,
    )


def test_slide_artifact_hard_denylist_rejects_variants() -> None:
    for title in (
        "Agenda",
        "Themen",
        "Gliederung",
        "Inhaltsverzeichnis",
        "Outline",
        "Recap",
        "Vielen-Dank",
        "Danke",
        "Fragen?",
        "Q&A",
        "Kontakt",
        "Ueber mich",
        "About",
        "Disclaimer",
    ):
        assert is_slide_artifact(_topic(title=title), document_page_count=100)


def test_substantive_titles_are_not_artifacts() -> None:
    assert not is_slide_artifact(_topic(title="process discovery"), document_page_count=100)
    assert not is_slide_artifact(_topic(title="conformance checking"), document_page_count=100)


def test_context_aware_reference_titles_filter_only_end_matter() -> None:
    end_matter = _topic(title="Literatur", summary="Dieses Thema behandelt Literatur und verweist knapp auf Quellen.", page_start=92, page_end=93)
    assert is_slide_artifact(end_matter, document_page_count=100, chunk_text_by_id={"c1": "short"})

    substantive = _topic(
        title="Literaturrecherche",
        summary="Literaturrecherche teaches how sources are found, assessed, and synthesized into a scientific argument.",
        page_start=30,
        chunk_ids=["c1", "c2", "c3", "c4", "c5"],
    )
    chunk_text_by_id = {f"c{index}": "substantive evidence text " * 20 for index in range(1, 6)}
    assert not is_slide_artifact(substantive, document_page_count=100, chunk_text_by_id=chunk_text_by_id)


def test_generic_summary_detection() -> None:
    assert is_generic_summary("Dieses Thema behandelt Process Discovery.", "Process Discovery")
    assert is_generic_summary("In diesem Abschnitt geht es um Process Discovery.", "Process Discovery")
    assert is_generic_summary("This topic discusses Process Discovery.", "Process Discovery")
    assert is_generic_summary("This section explains Process Discovery.", "Process Discovery")
    assert is_generic_summary("Es wird erläutert dass Process Discovery wichtig ist.", "Process Discovery")
    assert is_generic_summary("Process Discovery: Process Discovery is explained in this section.", "Process Discovery")
    assert not is_generic_summary(
        "Process discovery explains how event logs are transformed into process models and interpreted for improvement.",
        "Process Discovery",
    )


def test_low_content_uses_summary_genericity_and_evidence_budget() -> None:
    short_summary = _topic()
    object.__setattr__(short_summary, "summary", "Too short.")
    assert is_low_content_topic(short_summary, {"c1": "x" * 300})
    assert is_low_content_topic(_topic(), {"c1": "too short"})
    assert not is_low_content_topic(_topic(), {"c1": "x" * 120})
    assert is_low_content_topic(
        _topic(summary="This topic discusses Process Discovery and repeats the heading without adding substance."),
        {"c1": "x" * 120},
    )
    assert not is_low_content_topic(_topic(), {"c1": "x" * 300})


def test_filter_candidates_reports_rejections_and_keeps_attached_items() -> None:
    kept = _topic(title="Process Discovery", chunk_ids=["c1"])
    artifact = _topic(title="Agenda", chunk_ids=["c2"])
    low_content = _topic(title="Conformance Checking", chunk_ids=["c3"])
    generic = _topic(
        title="Performance Analysis",
        summary="This topic discusses Performance Analysis and repeats the heading without adding substance.",
        chunk_ids=["c4"],
    )

    kept_topics, kept_concepts, kept_objectives, rejections = filter_candidates(
        [kept, artifact, low_content, generic],
        [_concept("Process Discovery"), _concept("Agenda")],
        [_objective("Process Discovery"), _objective("Agenda")],
        FilterMeta(),
        {"c1": "x" * 300, "c2": "x" * 300, "c3": "short", "c4": "x" * 120},
    )

    assert kept_topics == [kept]
    assert [concept.topic_title for concept in kept_concepts] == ["Process Discovery"]
    assert [objective.topic_title for objective in kept_objectives] == ["Process Discovery"]
    assert {item["reason"] for item in rejections} == {"artifact", "low_content", "generic_summary"}
    assert all("evidence_chars" in item for item in rejections)
