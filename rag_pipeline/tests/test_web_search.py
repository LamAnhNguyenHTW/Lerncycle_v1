from __future__ import annotations

import re

from rag_pipeline.web_search import sanitize_web_query, search_web


class FakeTavilyClient:
    def __init__(self, results=None, raises: Exception | None = None) -> None:
        self.results = results if results is not None else []
        self.raises = raises
        self.calls = []

    def search(self, query: str, max_results: int):
        self.calls.append({"query": query, "max_results": max_results})
        if self.raises:
            raise self.raises
        return {"results": self.results}


def test_search_web_with_mocked_tavily_client_returns_normalized_results() -> None:
    client = FakeTavilyClient(
        [{"url": "https://example.com/a", "title": "Title", "content": "Content", "score": 0.9}]
    )

    outcome = search_web("query", top_k=5, api_key="key", client=client)

    assert outcome.error_type is None
    assert outcome.result_count == 1
    assert outcome.results[0]["source_type"] == "web"
    assert outcome.results[0]["metadata"]["url"] == "https://example.com/a"


def test_search_web_chunk_id_is_web_prefix_plus_16_char_hex_from_url() -> None:
    outcome = search_web(
        "query",
        api_key="key",
        client=FakeTavilyClient([{"url": "https://example.com/a", "title": "Title"}]),
    )

    assert re.fullmatch(r"web:[0-9a-f]{16}", outcome.results[0]["chunk_id"])


def test_search_web_empty_provider_results_returns_empty_list() -> None:
    outcome = search_web("query", api_key="key", client=FakeTavilyClient([]))

    assert outcome.results == []
    assert outcome.error_type == "empty_results"


def test_search_web_provider_exception_returns_empty_list() -> None:
    outcome = search_web("query", api_key="key", client=FakeTavilyClient(raises=RuntimeError("secret")))

    assert outcome.results == []
    assert outcome.error_type == "provider_error"


def test_search_web_missing_api_key_returns_empty_list() -> None:
    outcome = search_web("query", api_key=None)

    assert outcome.results == []
    assert outcome.error_type == "missing_api_key"


def test_search_web_top_k_limits_results() -> None:
    outcome = search_web(
        "query",
        top_k=1,
        api_key="key",
        client=FakeTavilyClient([
            {"url": "https://example.com/a", "title": "A"},
            {"url": "https://example.com/b", "title": "B"},
        ]),
    )

    assert len(outcome.results) == 1


def test_search_web_skips_results_without_url() -> None:
    outcome = search_web(
        "query",
        api_key="key",
        client=FakeTavilyClient([
            {"title": "Missing"},
            {"url": "https://example.com/b", "title": "B"},
        ]),
    )

    assert len(outcome.results) == 1
    assert outcome.results[0]["title"] == "B"


def test_search_web_default_score_is_0_5_when_missing() -> None:
    outcome = search_web(
        "query",
        api_key="key",
        client=FakeTavilyClient([{"url": "https://example.com/a", "title": "A"}]),
    )

    assert outcome.results[0]["score"] == 0.5


def test_search_web_fallback_text_uses_title_when_content_missing() -> None:
    outcome = search_web(
        "query",
        api_key="key",
        client=FakeTavilyClient([{"url": "https://example.com/a", "title": "A"}]),
    )

    assert outcome.results[0]["text"] == "A"


def test_sanitize_web_query_preserves_normal_german_query() -> None:
    assert sanitize_web_query("Was ist aktuell neu bei OpenAI?") == "Was ist aktuell neu bei OpenAI?"


def test_sanitize_web_query_preserves_normal_english_query() -> None:
    assert sanitize_web_query("What is new in OpenAI Agents SDK?") == "What is new in OpenAI Agents SDK?"


def test_sanitize_web_query_caps_long_query() -> None:
    assert len(sanitize_web_query("x" * 100, max_chars=10)) == 10


def test_sanitize_web_query_collapses_whitespace() -> None:
    assert sanitize_web_query("a\n\n   b\tc") == "a b c"


def test_sanitize_web_query_empty_input_returns_empty_string() -> None:
    assert sanitize_web_query("   ") == ""
