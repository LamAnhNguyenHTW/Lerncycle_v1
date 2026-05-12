"""Keyword-based chat memory intent detection."""

from __future__ import annotations


MEMORY_INTENT_PHRASES = (
    "was hatten wir",
    "was haben wir besprochen",
    "vorhin",
    "eben",
    "nochmal wie",
    "wie hast du es erklärt",
    "wie hast du es erklaert",
    "was habe ich gelernt",
    "was habe ich nicht verstanden",
    "offene fragen",
    "woran soll ich weiterlernen",
    "fass unseren lernstand zusammen",
    "what did we discuss",
    "what have i learned",
    "what did i struggle with",
    "open questions",
    "what should i review",
    "as before",
    "like earlier",
    "explain it like earlier",
)


def detect_memory_intent(
    query: str,
    recent_messages: list[dict] | None = None,
) -> bool:
    """Return True when a query asks about prior session context."""
    normalized = " ".join(query.lower().split())
    return any(phrase in normalized for phrase in MEMORY_INTENT_PHRASES)
