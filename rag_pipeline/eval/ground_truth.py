"""Ground-truth query set schema for retrieval evaluation.

The ground truth is deliberately *chunking-strategy independent*: relevance is
labeled by source id, page, and phrase, never by chunk id. The same query file
is therefore reusable across the fixed-size / Docling / Docling+refinement
collections, which is required for a fair chunking comparison.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


LEGACY_QUESTION_TYPES = {
    "factual",       # answer sits on a single page / short span
    "contextual",    # answer spans paragraphs or sections
    "table",         # answer is in a table or list
    "relational",    # relationship between concepts (graph-leaning)
    "cross_document",  # answer draws on more than one source
}
PILOT_QUESTION_TYPES = {
    "fact",
    "semantic_paraphrase",
    "terminology",
    "contextual",
    "relational",
    "multi_hop",
    "table_or_figure",
    "unanswerable",
}
QUESTION_TYPES = LEGACY_QUESTION_TYPES | PILOT_QUESTION_TYPES

LANGUAGES = {"de", "en"}
DIFFICULTIES = {"easy", "medium", "hard"}
PILOT_REQUIRED_FIELDS = {
    "query_id",
    "question",
    "language",
    "question_type",
    "expected_source_ids",
    "expected_pages",
    "expected_phrases",
    "reference_answer",
    "relevance_notes",
    "difficulty",
    "requires_multiple_chunks",
    "requires_multiple_documents",
    "extraction_risk",
    "manual_review_required",
}
MANIFEST_REQUIRED_FIELDS = {
    "source_id",
    "pdf_path",
    "title",
    "language",
    "source_type",
    "page_count",
}
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "credential",
    "password",
    "secret",
    "service_role",
    "token",
)


@dataclass(frozen=True)
class GroundTruthQuery:
    """One labeled evaluation query.

    Attributes:
        query_id: Stable unique id used in result files.
        question: The natural-language query sent to retrieval.
        language: 'de' or 'en'; enables per-language breakdowns.
        question_type: One of QUESTION_TYPES; enables per-type breakdowns.
        expected_source_ids: Source ids (pdf/note/annotation) that contain the
            answer. Any listed id counts as relevant.
        expected_pages: 1-based page numbers that contain the answer. Any listed
            page counts as relevant. Converted to 0-based page_index at match time.
        expected_phrases: Substrings that a relevant chunk should contain
            (case-insensitive). Any listed phrase counts as relevant.
        reference_answer: Gold answer for later answer-quality / LLM-judge runs;
            not used by retrieval metrics.
        notes: Optional free-text annotation (e.g. ambiguity, source rationale).
    """

    query_id: str
    question: str
    language: str
    question_type: str
    expected_source_ids: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    expected_phrases: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    notes: str | None = None
    relevance_notes: str | None = None
    difficulty: str | None = None
    requires_multiple_chunks: bool | None = None
    requires_multiple_documents: bool | None = None
    extraction_risk: bool | None = None
    manual_review_required: bool | None = None

    def has_relevance_labels(self) -> bool:
        """True when at least one relevance signal is present."""
        return bool(
            self.expected_source_ids
            or self.expected_pages
            or self.expected_phrases
        )


def _require(data: dict[str, Any], key: str, query_id: str) -> Any:
    if key not in data:
        raise ValueError(f"Query '{query_id}': missing required field '{key}'")
    return data[key]


def parse_query(data: dict[str, Any]) -> GroundTruthQuery:
    """Validate and build one GroundTruthQuery from a raw dict.

    Raises:
        ValueError: If required fields are missing or values are out of range.
    """
    query_id = str(data.get("query_id") or "")
    if not query_id:
        raise ValueError("Every query needs a non-empty 'query_id'.")

    question = str(_require(data, "question", query_id)).strip()
    if not question:
        raise ValueError(f"Query '{query_id}': 'question' must be non-empty.")

    language = str(_require(data, "language", query_id))
    if language not in LANGUAGES:
        raise ValueError(
            f"Query '{query_id}': language '{language}' not in {sorted(LANGUAGES)}"
        )

    question_type = str(_require(data, "question_type", query_id))
    if question_type not in QUESTION_TYPES:
        raise ValueError(
            f"Query '{query_id}': question_type '{question_type}' not in "
            f"{sorted(QUESTION_TYPES)}"
        )

    expected_pages = [int(page) for page in data.get("expected_pages", [])]
    for page in expected_pages:
        if page < 1:
            raise ValueError(
                f"Query '{query_id}': expected_pages are 1-based and must be >= 1."
            )

    query = GroundTruthQuery(
        query_id=query_id,
        question=question,
        language=language,
        question_type=question_type,
        expected_source_ids=[str(sid) for sid in data.get("expected_source_ids", [])],
        expected_pages=expected_pages,
        expected_phrases=[str(phrase) for phrase in data.get("expected_phrases", [])],
        reference_answer=(
            str(data["reference_answer"]) if data.get("reference_answer") else None
        ),
        notes=str(data["notes"]) if data.get("notes") else None,
        relevance_notes=(
            str(data["relevance_notes"]) if data.get("relevance_notes") else None
        ),
        difficulty=str(data["difficulty"]) if data.get("difficulty") else None,
        requires_multiple_chunks=data.get("requires_multiple_chunks"),
        requires_multiple_documents=data.get("requires_multiple_documents"),
        extraction_risk=data.get("extraction_risk"),
        manual_review_required=data.get("manual_review_required"),
    )
    if not query.has_relevance_labels() and query.question_type != "unanswerable":
        raise ValueError(
            f"Query '{query_id}': needs at least one of expected_source_ids, "
            "expected_pages, or expected_phrases."
        )
    return query


def load_ground_truth(path: str | Path) -> list[GroundTruthQuery]:
    """Load and validate a ground-truth JSON file.

    The file may be a list of query objects, or an object with a 'queries' key.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["queries"] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Ground-truth file must be a list or {'queries': [...]}.")
    queries = [parse_query(item) for item in items]
    _assert_unique_ids(queries)
    return queries


def _assert_unique_ids(queries: list[GroundTruthQuery]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            duplicates.add(query.query_id)
        seen.add(query.query_id)
    if duplicates:
        raise ValueError(f"Duplicate query_id(s): {sorted(duplicates)}")


def validate_pilot_ground_truth(
    query_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_pdf_page_counts: bool = True,
) -> list[GroundTruthQuery]:
    """Validate the strict pilot schema against its local corpus manifest.

    Args:
        query_path: Pilot JSON file containing a top-level queries list.
        manifest_path: Corpus manifest with stable source IDs and page counts.
        verify_pdf_page_counts: Read each local PDF and compare its actual page
            count with the manifest. Tests may disable this for dummy fixtures.

    Returns:
        Parsed, validated pilot queries.

    Raises:
        ValueError: If schema, security, source, file, or page checks fail.
    """
    query_file = Path(query_path)
    manifest_file = Path(manifest_path)
    raw_queries = json.loads(query_file.read_text(encoding="utf-8"))
    raw_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _assert_safe_json(raw_queries)
    _assert_safe_json(raw_manifest)

    sources = _validate_manifest(
        raw_manifest,
        manifest_file,
        verify_pdf_page_counts=verify_pdf_page_counts,
    )
    items = raw_queries.get("queries") if isinstance(raw_queries, dict) else None
    if not isinstance(items, list):
        raise ValueError("Pilot ground truth must be an object with a queries list.")

    queries: list[GroundTruthQuery] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every pilot query must be a JSON object.")
        query_id = str(item.get("query_id") or "<unknown>")
        missing = sorted(PILOT_REQUIRED_FIELDS - set(item))
        if missing:
            raise ValueError(
                f"Query '{query_id}': missing required field(s): {missing}"
            )
        _validate_pilot_item(item, sources)
        queries.append(parse_query(item))
    _assert_unique_ids(queries)
    return queries


def _validate_manifest(
    raw_manifest: Any,
    manifest_path: Path,
    *,
    verify_pdf_page_counts: bool,
) -> dict[str, int]:
    if not isinstance(raw_manifest, dict):
        raise ValueError("Corpus manifest must be a JSON object.")
    documents = raw_manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus manifest needs a non-empty documents list.")

    root = manifest_path.resolve().parent
    sources: dict[str, int] = {}
    listed_files: set[Path] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Every manifest document must be a JSON object.")
        source_id = str(document.get("source_id") or "<unknown>")
        missing = sorted(MANIFEST_REQUIRED_FIELDS - set(document))
        if missing:
            raise ValueError(
                f"Manifest source '{source_id}': missing field(s): {missing}"
            )
        if source_id in sources:
            raise ValueError(f"Duplicate manifest source_id: '{source_id}'.")

        relative_path = str(document["pdf_path"])
        if _is_absolute_local_path(relative_path):
            raise ValueError(
                f"Manifest source '{source_id}': pdf_path must be relative."
            )
        pdf_path = (root / relative_path).resolve()
        if pdf_path.parent != root or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Manifest source '{source_id}': pdf_path must name a corpus PDF."
            )
        if not pdf_path.is_file():
            raise ValueError(
                f"Manifest source '{source_id}': PDF file does not exist."
            )
        page_count = document["page_count"]
        if isinstance(page_count, bool) or not isinstance(page_count, int):
            raise ValueError(
                f"Manifest source '{source_id}': page_count must be an integer."
            )
        if page_count < 1:
            raise ValueError(
                f"Manifest source '{source_id}': page_count must be positive."
            )
        if verify_pdf_page_counts:
            actual_count = _pdf_page_count(pdf_path)
            if actual_count != page_count:
                raise ValueError(
                    f"Manifest source '{source_id}': page_count does not match PDF."
                )
        sources[source_id] = page_count
        listed_files.add(pdf_path)

    corpus_files = {path.resolve() for path in root.glob("*.pdf")}
    unknown_files = sorted(path.name for path in corpus_files - listed_files)
    if unknown_files:
        raise ValueError(f"Corpus contains unknown PDF file(s): {unknown_files}")
    return sources


def _validate_pilot_item(item: dict[str, Any], sources: dict[str, int]) -> None:
    query_id = str(item["query_id"])
    if item["question_type"] not in PILOT_QUESTION_TYPES:
        raise ValueError(
            f"Query '{query_id}': invalid pilot question_type "
            f"'{item['question_type']}'."
        )
    if item["difficulty"] not in DIFFICULTIES:
        raise ValueError(
            f"Query '{query_id}': invalid difficulty '{item['difficulty']}'."
        )
    for field_name in (
        "requires_multiple_chunks",
        "requires_multiple_documents",
        "extraction_risk",
        "manual_review_required",
    ):
        if not isinstance(item[field_name], bool):
            raise ValueError(
                f"Query '{query_id}': '{field_name}' must be boolean."
            )

    for field_name in (
        "expected_source_ids",
        "expected_pages",
        "expected_phrases",
    ):
        if not isinstance(item[field_name], list):
            raise ValueError(
                f"Query '{query_id}': '{field_name}' must be a list."
            )
    for field_name in (
        "question",
        "reference_answer",
        "relevance_notes",
    ):
        if not isinstance(item[field_name], str) or not item[field_name].strip():
            raise ValueError(
                f"Query '{query_id}': '{field_name}' must be non-empty."
            )

    source_ids = item["expected_source_ids"]
    unknown_sources = sorted(set(source_ids) - set(sources))
    if unknown_sources:
        raise ValueError(
            f"Query '{query_id}': unknown source_id(s): {unknown_sources}"
        )

    if item["question_type"] == "unanswerable":
        if source_ids or item["expected_pages"] or item["expected_phrases"]:
            raise ValueError(
                f"Query '{query_id}': unanswerable labels must all be empty."
            )
        if (
            item["requires_multiple_chunks"]
            or item["requires_multiple_documents"]
        ):
            raise ValueError(
                f"Query '{query_id}': unanswerable query cannot require sources."
            )
        return

    if not source_ids:
        raise ValueError(
            f"Query '{query_id}': answerable query needs expected_source_ids."
        )
    if not item["expected_pages"]:
        raise ValueError(
            f"Query '{query_id}': answerable query needs expected_pages."
        )
    if item["requires_multiple_documents"] != (len(set(source_ids)) > 1):
        raise ValueError(
            f"Query '{query_id}': requires_multiple_documents is inconsistent."
        )
    for page in item["expected_pages"]:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(
                f"Query '{query_id}': expected_pages must be positive integers."
            )
        for source_id in source_ids:
            if page > sources[source_id]:
                raise ValueError(
                    f"Query '{query_id}': page {page} is outside source "
                    f"'{source_id}'."
                )


def _pdf_page_count(pdf_path: Path) -> int:
    import pypdfium2

    document = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return len(document)
    finally:
        document.close()


def _assert_safe_json(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for raw_key, nested_value in value.items():
            key = str(raw_key)
            normalized_key = key.lower().replace("-", "_")
            if any(part in normalized_key for part in _SECRET_KEY_FRAGMENTS):
                raise ValueError(
                    f"Ground-truth JSON contains a secret-like key at {path}."
                )
            _assert_safe_json(nested_value, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _assert_safe_json(nested_value, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc and (
            parsed.username is not None or parsed.password is not None
        ):
            raise ValueError(
                f"Ground-truth JSON contains a credentialed URL at {path}."
            )
        if _is_absolute_local_path(value):
            raise ValueError(
                f"Ground-truth JSON contains an absolute local path at {path}."
            )


def _is_absolute_local_path(value: str) -> bool:
    if PureWindowsPath(value).is_absolute():
        return True
    return bool(
        value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or value.lower().startswith("file:")
    )


def is_hit_relevant(hit: dict[str, Any], query: GroundTruthQuery) -> bool:
    """Return whether a retrieval hit satisfies the query's relevance labels.

    Source and page are primary labels. When both are present, both must match;
    otherwise a chunk from the right document but the wrong page would inflate
    every metric. Expected phrases are supplementary and serve as the fallback
    only when neither source nor page labels are available.
    """
    source_id = hit.get("source_id")
    page_index = hit.get("page_index")
    has_sources = bool(query.expected_source_ids)
    has_pages = bool(query.expected_pages)
    source_matches = (
        source_id is not None
        and str(source_id) in set(query.expected_source_ids)
    )
    page_matches = (
        page_index is not None
        and int(page_index) + 1 in set(query.expected_pages)
    )

    if has_sources and has_pages:
        return source_matches and page_matches
    if has_sources:
        return source_matches
    if has_pages:
        return page_matches

    text = str(hit.get("text") or "").lower()
    if text:
        for phrase in query.expected_phrases:
            if phrase.strip() and phrase.lower() in text:
                return True
    return False
