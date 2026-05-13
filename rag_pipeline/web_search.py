"""Optional live web search retrieval provider integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import socket
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError


logger = logging.getLogger(__name__)

WEB_ERROR_TYPES = {
    "missing_api_key",
    "provider_error",
    "timeout",
    "empty_results",
    "invalid_query",
}


@dataclass(frozen=True)
class WebSearchOutcome:
    """Structured web search result with safe metadata."""

    results: list[dict[str, Any]]
    provider: str
    result_count: int
    error_type: str | None = None


def sanitize_web_query(query: str, max_chars: int = 300) -> str:
    """Trim, collapse whitespace, and cap a query before provider calls."""
    collapsed = re.sub(r"\s+", " ", str(query or "")).strip()
    return collapsed[:max_chars]


def search_web(
    query: str,
    top_k: int = 5,
    provider: str = "tavily",
    api_key: str | None = None,
    timeout_seconds: int = 15,
    max_query_chars: int = 300,
    client: Any | None = None,
) -> WebSearchOutcome:
    """Run optional web search and return normalized chunk-like results."""
    safe_provider = provider or "tavily"
    sanitized = sanitize_web_query(query, max_query_chars)
    if not sanitized:
        return WebSearchOutcome([], safe_provider, 0, "invalid_query")
    if not api_key:
        return WebSearchOutcome([], safe_provider, 0, "missing_api_key")
    if safe_provider != "tavily":
        return WebSearchOutcome([], safe_provider, 0, "provider_error")

    try:
        active_client = client or _build_tavily_client(api_key, timeout_seconds)
        payload = active_client.search(sanitized, max_results=top_k)
        raw_results = payload.get("results") if isinstance(payload, dict) else []
        results = _normalize_tavily_results(raw_results or [], top_k)
        if not results:
            return WebSearchOutcome([], safe_provider, 0, "empty_results")
        return WebSearchOutcome(results, safe_provider, len(results), None)
    except TimeoutError:
        logger.warning("Web search failed", extra={"error_type": "timeout", "provider": safe_provider})
        return WebSearchOutcome([], safe_provider, 0, "timeout")
    except (socket.timeout, TimeoutError):
        logger.warning("Web search failed", extra={"error_type": "timeout", "provider": safe_provider})
        return WebSearchOutcome([], safe_provider, 0, "timeout")
    except Exception:
        logger.warning("Web search failed", extra={"error_type": "provider_error", "provider": safe_provider})
        return WebSearchOutcome([], safe_provider, 0, "provider_error")


class _TavilyClient:
    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> dict[str, Any]:
        body = json.dumps(
            {
                "query": query,
                "max_results": max_results,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            "https://api.tavily.com/search",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _build_tavily_client(api_key: str, timeout_seconds: int) -> _TavilyClient:
    """Create a minimal Tavily HTTP client."""
    return _TavilyClient(api_key, timeout_seconds)


def _normalize_tavily_results(raw_results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    normalized = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for index, item in enumerate(raw_results, start=1):
        if len(normalized) >= top_k:
            break
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        title = _first_text(item.get("title"), _domain(url), "Web Source") or "Web Source"
        text = _first_text(item.get("content"), item.get("snippet"), title) or title
        chunk_id = "web:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        score = item.get("score")
        normalized.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": float(score) if isinstance(score, int | float) else 0.5,
                "source_type": "web",
                "source_id": chunk_id,
                "title": title,
                "heading": None,
                "page_index": None,
                "pdf_id": None,
                "metadata": {
                    "url": url,
                    "title": title,
                    "provider": "tavily",
                    "published_date": item.get("published_date"),
                    "retrieved_at": retrieved_at,
                    "rank": index,
                },
            }
        )
    return normalized


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _domain(url: str) -> str | None:
    match = re.match(r"https?://([^/]+)", url)
    return match.group(1) if match else None
