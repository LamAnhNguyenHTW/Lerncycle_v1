from __future__ import annotations

from rag_pipeline.rag_answer import (
    CONVERSATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_with_rag,
    rewrite_query_for_retrieval,
)
from rag_pipeline.reranker import LlmReranker
from rag_pipeline.web_search import WebSearchOutcome
from rag_pipeline.intent_classifier import RetrievalIntent
from rag_pipeline.retrieval_plan import PlanExecutionOutcome
from rag_pipeline.retrieval_plan import RetrievalPlan
from rag_pipeline.retrieval_plan import RetrievalPlanStep


class FakeLlmClient:
    def __init__(self, answer: str = "Antwort") -> None:
        self.answer = answer
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.answer


class RaisingLlmClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        raise RuntimeError("rewrite failed")


class SpyReranker:
    def __init__(self, output: list[dict] | None = None, raises: bool = False) -> None:
        self.output = output
        self.raises = raises
        self.calls = []

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        self.calls.append({"query": query, "results": results, "top_k": top_k})
        if self.raises:
            raise RuntimeError("rerank failed")
        return self.output if self.output is not None else results[:top_k]


class MemoryAwareRetrieval:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("source_types") == ["chat_memory"]:
            return [_result_with(9, "Earlier we discussed Process Mining basics.") | {
                "source_type": "chat_memory",
                "source_id": kwargs["source_ids"][0],
                "title": "Chat Memory",
                "heading": "Learning history",
                "metadata": {"session_id": kwargs["source_ids"][0], "memory_kind": "rolling_summary"},
            }]
        return [_result_with(1, "Material chunk")]


class FakeGraphStore:
    def __init__(self, raises: bool = False) -> None:
        self.raises = raises
        self.calls = []

    def search_concepts(self, **kwargs):
        self.calls.append(("search", kwargs))
        if self.raises:
            raise RuntimeError("graph failed")
        return [{"name": "Process Mining", "normalized_name": "process mining"}]

    def get_neighborhood(self, **kwargs):
        self.calls.append(("neighborhood", kwargs))
        return {
            "relationships": [
                {
                    "source": "Process Mining",
                    "target": "Event Logs",
                    "relation_type": "uses",
                    "description": "uses event logs as input",
                    "chunk_id": "chunk-graph-1",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                }
            ]
        }


def _web_result() -> dict:
    return {
        "chunk_id": "web:1234567890abcdef",
        "text": "Current web information.",
        "score": 0.7,
        "source_type": "web",
        "source_id": "web:1234567890abcdef",
        "title": "Web Source",
        "heading": None,
        "page_index": None,
        "metadata": {"url": "https://example.com", "provider": "tavily"},
    }


def _intent(**overrides) -> RetrievalIntent:
    payload = {
        "question_type": "current_external_info",
        "needs_pdf": True,
        "needs_notes": True,
        "needs_annotations": True,
        "needs_chat_memory": False,
        "needs_graph": False,
        "needs_web": False,
        "confidence": 0.8,
        "reasoning_summary": "test",
    }
    payload.update(overrides)
    return RetrievalIntent(**payload)


def _result() -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": "Process Mining verbindet Data Science und Process Science.",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "page_index": 9,
        "title": "GPAA.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA.pdf"},
    }


def _result_with(index: int, text: str) -> dict:
    result = _result()
    result["chunk_id"] = f"chunk-{index}"
    result["text"] = text
    result["score"] = 1.0 / index
    return result


def test_rewrite_query_returns_original_without_recent_messages() -> None:
    llm = FakeLlmClient("Process Mining Definition")

    rewritten = rewrite_query_for_retrieval("Was ist Process Mining?", None, llm)

    assert rewritten == "Was ist Process Mining?"
    assert llm.calls == []


def test_rewrite_query_uses_recent_messages_for_vague_followup() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt Definition Event Logs")

    rewritten = rewrite_query_for_retrieval(
        "Kannst du mir das einfacher erklären?",
        [
            {"role": "user", "content": "Was ist Process Mining?"},
            {"role": "assistant", "content": "Process Mining analysiert Event Logs."},
        ],
        llm,
    )

    assert rewritten == "Process Mining einfach erklärt Definition Event Logs"
    assert "Was ist Process Mining?" in llm.calls[0]["user_prompt"]
    assert "Kannst du mir das einfacher erklären?" in llm.calls[0]["user_prompt"]


def test_rewrite_query_keeps_standalone_query_mostly_unchanged() -> None:
    llm = FakeLlmClient("Andere Antwort")

    rewritten = rewrite_query_for_retrieval(
        "Was ist Process Mining?",
        [{"role": "assistant", "content": "Vorherige Antwort"}],
        llm,
    )

    assert rewritten == "Was ist Process Mining?"
    assert llm.calls == []


def test_rewrite_query_falls_back_to_original_on_error() -> None:
    llm = RaisingLlmClient()

    rewritten = rewrite_query_for_retrieval(
        "Erklär das einfacher.",
        [{"role": "assistant", "content": "Process Mining analysiert Event Logs."}],
        llm,
    )

    assert rewritten == "Erklär das einfacher."
    assert len(llm.calls) == 1


def test_answer_with_rag_calls_hybrid_retrieval_with_user_id() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag("Was ist Process Mining?", "user-1", llm_client=FakeLlmClient(), retrieval_fn=retrieval_fn)

    assert calls[0]["user_id"] == "user-1"
    assert calls[0]["query"] == "Was ist Process Mining?"


def test_answer_with_rag_builds_context() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    assert "Type: PDF" in llm.calls[0]["user_prompt"]
    assert "Page: 10" in llm.calls[0]["user_prompt"]


def test_answer_with_rag_returns_answer_and_sources() -> None:
    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient("Eine Antwort"),
        retrieval_fn=lambda **_: [_result()],
    )

    assert response["answer"] == "Eine Antwort"
    assert response["sources"][0]["chunk_id"] == "chunk-1"


def test_web_search_not_used_by_default() -> None:
    calls = []

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        web_search_fn=lambda **kwargs: calls.append(kwargs) or WebSearchOutcome([], "tavily", 0),
    )

    assert calls == []


def test_web_search_used_when_web_mode_on_and_enabled() -> None:
    calls = []

    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        web_mode="on",
        web_search_enabled=True,
        web_search_fn=lambda **kwargs: calls.append(kwargs) or WebSearchOutcome([_web_result()], "tavily", 1),
    )

    assert calls
    assert any(source["source_type"] == "web" for source in response["sources"])
    assert response["web_search"]["used"] is True


def test_answer_with_rag_does_not_plan_when_disabled() -> None:
    planner_calls = []

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent=_intent(needs_pdf=True),
        retrieval_planner_enabled=False,
        retrieval_planner_fn=lambda **kwargs: planner_calls.append(kwargs),
    )

    assert planner_calls == []


def test_answer_with_rag_executes_plan_results() -> None:
    plan = RetrievalPlan(
        question_type="document_grounded",
        steps=[RetrievalPlanStep(tool="search_pdf_chunks", query="Frage", top_k=1)],
    )

    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [],
        intent=_intent(needs_pdf=True),
        retrieval_planner_enabled=True,
        retrieval_planner_fn=lambda **_: plan,
        retrieval_plan_executor_fn=lambda **_: PlanExecutionOutcome([_result()], [{"tool": "search_pdf_chunks", "status": "enabled", "result_count": 1, "error_type": None}], 1),
    )

    assert response["sources"][0]["source_type"] == "pdf"
    assert response["retrieval_plan"]["planner_used"] is True
    assert response["retrieval_plan"]["steps"][0]["result_count"] == 1


def test_answer_with_rag_planner_failure_falls_back() -> None:
    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent=_intent(needs_pdf=True),
        retrieval_planner_enabled=True,
        retrieval_planner_fn=lambda **_: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    assert response["sources"][0]["chunk_id"] == "chunk-1"
    assert response["retrieval_plan"]["fallback_used"] is True


def test_web_search_provider_error_does_not_fail_answer() -> None:
    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        web_mode="on",
        web_search_enabled=True,
        web_search_fn=lambda **_: WebSearchOutcome([], "tavily", 0, "provider_error"),
    )

    assert response["answer"] == "Antwort"
    assert response["web_search"]["error_type"] == "provider_error"


def test_answer_with_rag_does_not_classify_when_disabled() -> None:
    calls = []

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent_classifier_fn=lambda **kwargs: calls.append(kwargs) or _intent(),
    )

    assert calls == []


def test_answer_with_rag_classifies_when_enabled() -> None:
    calls = []

    response = answer_with_rag(
        "Was ist aktuell neu?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent_classifier_enabled=True,
        intent_classifier_fn=lambda **kwargs: calls.append(kwargs) or _intent(needs_web=True),
        web_search_enabled=False,
    )

    assert calls
    assert response["intent"]["needs_web"] is True
    assert response["intent"]["web_skipped_reason"] == "disabled"


def test_intent_needs_web_enables_web_when_config_enabled() -> None:
    web_calls = []

    response = answer_with_rag(
        "Was ist aktuell neu?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent=_intent(needs_web=True),
        web_search_enabled=True,
        web_search_fn=lambda **kwargs: web_calls.append(kwargs) or WebSearchOutcome([_web_result()], "tavily", 1),
    )

    assert web_calls
    assert response["web_search"]["used"] is True


def test_intent_needs_memory_requires_session_id() -> None:
    retrieval_calls = []

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **kwargs: retrieval_calls.append(kwargs) or [_result()],
        intent=_intent(question_type="conversation_memory", needs_chat_memory=True),
        chat_memory_retrieval_enabled=True,
        session_id=None,
    )

    assert all(call.get("source_types") != ["chat_memory"] for call in retrieval_calls)


def test_intent_needs_graph_sets_metadata_but_does_not_force_graph_store() -> None:
    response = answer_with_rag(
        "Zusammenhang?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        intent=_intent(question_type="concept_relationship", needs_graph=True),
        graph_retrieval_enabled=False,
    )

    assert response["intent"]["graph_requested"] is True
    assert response["intent"]["graph_available"] is False


def test_prompt_distinguishes_web_sources_from_internal_sources() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        web_mode="on",
        web_search_enabled=True,
        web_search_fn=lambda **_: WebSearchOutcome([_web_result()], "tavily", 1),
    )

    assert "External web sources" in llm.calls[-1]["system_prompt"]


def test_answer_with_rag_handles_no_results() -> None:
    llm = FakeLlmClient()

    response = answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [])

    assert response["sources"] == []
    assert "keine passenden Quellen" in response["answer"]
    assert llm.calls == []


def test_answer_with_rag_does_not_call_llm_when_no_context() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [])

    assert len(llm.calls) == 0


def test_answer_with_rag_passes_source_types_to_retrieval() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Frage",
        "user-1",
        source_types=["note"],
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval_fn,
    )

    assert calls[0]["source_types"] == ["note"]


def test_answer_with_rag_uses_injected_llm_client() -> None:
    llm = FakeLlmClient("Custom")

    response = answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    assert response["answer"] == "Custom"
    assert len(llm.calls) == 1


def test_answer_with_rag_accepts_recent_messages() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Was ist Process Mining?"}],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert "Was ist Process Mining?" in llm.calls[0]["user_prompt"]


def test_answer_with_rag_includes_recent_messages_in_prompt() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[
            {"role": "user", "content": "Was ist Process Mining?"},
            {"role": "assistant", "content": "Process Mining analysiert Prozessdaten."},
        ],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    prompt = llm.calls[-1]["user_prompt"]
    assert "Recent conversation:" in prompt
    assert "User: Was ist Process Mining?" in prompt
    assert "Assistant: Process Mining analysiert Prozessdaten." in prompt
    assert "Retrieved context:" in prompt


def test_answer_with_rag_includes_context_summary_before_recent_messages() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Was ist Process Mining?"}],
        context_summary="Earlier summary text",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    prompt = llm.calls[-1]["user_prompt"]
    assert "[Conversation summary" in prompt
    assert "Earlier summary text" in prompt
    assert prompt.index("[Conversation summary") < prompt.index("Recent conversation:")


def test_answer_with_rag_omits_empty_context_summary_block() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Was ist Process Mining?"}],
        context_summary="   ",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    prompt = llm.calls[-1]["user_prompt"]
    assert "[Conversation summary" not in prompt
    assert "Recent conversation:" in prompt


def test_answer_with_rag_preserves_current_query_as_final_question() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Was ist Process Mining?"}],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert llm.calls[-1]["user_prompt"].rstrip().endswith("Current question:\nErklär das einfacher.")


def test_answer_with_rag_does_not_call_llm_when_no_retrieval_context_even_with_recent_messages() -> None:
    llm = FakeLlmClient()

    response = answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Was ist Process Mining?"}],
        llm_client=llm,
        retrieval_fn=lambda **_: [],
    )

    assert response["sources"] == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["system_prompt"].startswith("Rewrite the current user question")


def test_answer_with_rag_does_not_return_recent_messages_as_sources() -> None:
    response = answer_with_rag(
        "Erklär das einfacher.",
        "user-1",
        recent_messages=[{"role": "user", "content": "Chat-only detail"}],
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
    )

    assert all(source.get("source_type") != "chat_message" for source in response["sources"])
    assert "Chat-only detail" not in str(response["sources"])


def test_answer_with_rag_uses_rewritten_query_for_retrieval() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt Definition Event Logs")
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Kannst du mir das einfacher erklären?",
        "user-1",
        recent_messages=[{"role": "assistant", "content": "Process Mining analysiert Event Logs."}],
        llm_client=llm,
        retrieval_fn=retrieval_fn,
    )

    assert calls[0]["query"] == "Process Mining einfach erklärt Definition Event Logs"


def test_answer_with_rag_preserves_original_query_for_final_answer() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt Definition Event Logs")

    answer_with_rag(
        "Kannst du mir das einfacher erklären?",
        "user-1",
        recent_messages=[{"role": "assistant", "content": "Process Mining analysiert Event Logs."}],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert "Current question:\nKannst du mir das einfacher erklären?" in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_does_not_expose_rewritten_query() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt Definition Event Logs")

    response = answer_with_rag(
        "Kannst du mir das einfacher erklären?",
        "user-1",
        recent_messages=[{"role": "assistant", "content": "Process Mining analysiert Event Logs."}],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert "retrieval_query" not in response


def test_prompt_states_recent_conversation_is_for_continuity_only() -> None:
    prompt = CONVERSATION_SYSTEM_PROMPT.lower()

    assert "recent conversation" in prompt
    assert "continuity" in prompt
    assert "only" in prompt


def test_prompt_states_retrieved_context_is_source_of_truth() -> None:
    prompt = CONVERSATION_SYSTEM_PROMPT.lower()

    assert "retrieved context" in prompt
    assert "source of truth" in prompt


def test_prompt_instructs_simpler_shorter_answer_for_simplification_requests() -> None:
    prompt = CONVERSATION_SYSTEM_PROMPT.lower()

    assert "einfacher" in prompt
    assert "short sentences" in prompt
    assert "main idea" in prompt


def test_prompt_discourages_im_bereitgestellten_kontext_opener() -> None:
    assert 'do not begin your answer with "im bereitgestellten kontext"' in SYSTEM_PROMPT.lower()
    assert 'do not begin your answer with "im bereitgestellten kontext"' in CONVERSATION_SYSTEM_PROMPT.lower()


def test_answer_with_rag_skips_reranker_when_disabled() -> None:
    reranker = SpyReranker()

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        reranker=reranker,
        reranking_enabled=False,
    )

    assert reranker.calls == []


def test_answer_with_rag_uses_candidate_k_for_retrieval_when_reranking_enabled() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval_fn,
        reranker=SpyReranker(),
        reranking_enabled=True,
        reranking_candidate_k=22,
        reranking_top_k=5,
    )

    assert calls[0]["top_k"] == 22


def test_answer_with_rag_passes_results_to_reranker() -> None:
    results = [_result()]
    reranker = SpyReranker()

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: results,
        reranker=reranker,
        reranking_enabled=True,
    )

    assert reranker.calls[0]["results"] == results


def test_answer_with_rag_builds_context_from_reranked_results() -> None:
    llm = FakeLlmClient()
    reranked = [_result() | {"text": "Reranked chunk text"}]

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result() | {"text": "Original chunk text"}],
        reranker=SpyReranker(output=reranked),
        reranking_enabled=True,
    )

    assert "Reranked chunk text" in llm.calls[-1]["user_prompt"]
    assert "Original chunk text" not in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_falls_back_when_reranker_fails() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result() | {"text": "Original chunk text"}],
        reranker=SpyReranker(raises=True),
        reranking_enabled=True,
    )

    assert "Original chunk text" in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_preserves_scope_filters_with_reranking() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Frage",
        "user-1",
        source_types=["pdf"],
        pdf_ids=["pdf-1"],
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval_fn,
        reranker=SpyReranker(),
        reranking_enabled=True,
    )

    assert calls[0]["user_id"] == "user-1"
    assert calls[0]["source_types"] == ["pdf"]
    assert calls[0]["pdf_ids"] == ["pdf-1"]


def test_answer_with_rag_still_uses_original_query_for_final_answer() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt")

    answer_with_rag(
        "Kannst du mir das einfacher erklären?",
        "user-1",
        recent_messages=[{"role": "assistant", "content": "Process Mining"}],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        reranker=SpyReranker(),
        reranking_enabled=True,
    )

    assert "Current question:\nKannst du mir das einfacher erklären?" in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_uses_rewritten_query_for_retrieval_before_reranking() -> None:
    llm = FakeLlmClient("Process Mining einfach erklärt")
    retrieval_calls = []
    reranker = SpyReranker()

    def retrieval_fn(**kwargs):
        retrieval_calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Kannst du mir das einfacher erklären?",
        "user-1",
        recent_messages=[{"role": "assistant", "content": "Process Mining"}],
        llm_client=llm,
        retrieval_fn=retrieval_fn,
        reranker=reranker,
        reranking_enabled=True,
    )

    assert retrieval_calls[0]["query"] == "Process Mining einfach erklärt"
    assert reranker.calls[0]["query"] == "Process Mining einfach erklärt"
def test_answer_with_rag_accepts_llm_reranker() -> None:
    reranker = LlmReranker(
        llm_client=FakeLlmClient('[{"chunk_id":"chunk-1","score":0.9}]')
    )

    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient("Antwort"),
        retrieval_fn=lambda **_: [_result()],
        reranker=reranker,
        reranking_enabled=True,
    )

    assert response["answer"] == "Antwort"


def test_answer_with_rag_builds_context_from_llm_reranked_results() -> None:
    answer_llm = FakeLlmClient("Antwort")
    reranker = LlmReranker(
        llm_client=FakeLlmClient('[{"chunk_id":"chunk-2","score":0.9}]')
    )

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=answer_llm,
        retrieval_fn=lambda **_: [
            _result_with(1, "Original first chunk"),
            _result_with(2, "LLM preferred chunk"),
        ],
        reranker=reranker,
        reranking_enabled=True,
        reranking_top_k=2,
    )

    prompt = answer_llm.calls[-1]["user_prompt"]
    assert prompt.index("LLM preferred chunk") < prompt.index("Original first chunk")


def test_answer_with_rag_falls_back_when_llm_reranker_fails() -> None:
    answer_llm = FakeLlmClient("Antwort")
    reranker = LlmReranker(llm_client=FakeLlmClient("invalid json"))

    answer_with_rag(
        "Frage",
        "user-1",
        llm_client=answer_llm,
        retrieval_fn=lambda **_: [
            _result_with(1, "Original first chunk"),
            _result_with(2, "Original second chunk"),
        ],
        reranker=reranker,
        reranking_enabled=True,
        reranking_top_k=2,
    )

    prompt = answer_llm.calls[-1]["user_prompt"]
    assert prompt.index("Original first chunk") < prompt.index("Original second chunk")


def test_chat_memory_not_included_by_default() -> None:
    retrieval = MemoryAwareRetrieval()

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval,
        session_id="session-1",
    )

    assert len(retrieval.calls) == 1


def test_chat_memory_included_for_memory_intent_when_enabled() -> None:
    retrieval = MemoryAwareRetrieval()
    llm = FakeLlmClient()

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=llm,
        retrieval_fn=retrieval,
        session_id="session-1",
        chat_memory_retrieval_enabled=True,
    )

    assert retrieval.calls[1]["source_types"] == ["chat_memory"]
    assert retrieval.calls[1]["source_ids"] == ["session-1"]
    assert "Earlier we discussed" in llm.calls[-1]["user_prompt"]


def test_chat_memory_requires_session_id() -> None:
    retrieval = MemoryAwareRetrieval()

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval,
        chat_memory_retrieval_enabled=True,
    )

    assert len(retrieval.calls) == 1


def test_memory_retrieval_filters_by_current_session_id() -> None:
    retrieval = MemoryAwareRetrieval()

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval,
        session_id="session-current",
        chat_memory_retrieval_enabled=True,
    )

    assert retrieval.calls[1]["user_id"] == "user-1"
    assert retrieval.calls[1]["source_ids"] == ["session-current"]


def test_memory_retrieval_includes_related_memory_source_ids() -> None:
    retrieval = MemoryAwareRetrieval()

    answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval,
        session_id="session-current",
        memory_source_ids=["session-old"],
        chat_memory_retrieval_enabled=True,
    )

    assert retrieval.calls[1]["source_ids"] == ["session-current", "session-old"]


def test_answer_with_rag_skips_graph_when_disabled() -> None:
    graph_store = FakeGraphStore()

    answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        graph_store=graph_store,
        graph_retrieval_enabled=False,
    )

    assert graph_store.calls == []


def test_answer_with_rag_uses_graph_for_graph_intent() -> None:
    graph_store = FakeGraphStore()
    llm = FakeLlmClient()

    answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        graph_store=graph_store,
        graph_retrieval_enabled=True,
    )

    assert graph_store.calls
    assert "Knowledge Graph Context:" in llm.calls[-1]["user_prompt"]
    assert "Process Mining --uses--> Event Logs" in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_does_not_use_graph_for_normal_question_by_default() -> None:
    graph_store = FakeGraphStore()

    answer_with_rag(
        "Was ist Process Mining?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        graph_store=graph_store,
        graph_retrieval_enabled=True,
    )

    assert graph_store.calls == []


def test_answer_with_rag_combines_graph_and_text_context() -> None:
    graph_store = FakeGraphStore()
    llm = FakeLlmClient()

    answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        graph_store=graph_store,
        graph_retrieval_enabled=True,
    )

    prompt = llm.calls[-1]["user_prompt"]
    assert "Text Chunk Context:" in prompt
    assert "Knowledge Graph Context:" in prompt


def test_answer_with_rag_falls_back_when_graph_retrieval_fails() -> None:
    llm = FakeLlmClient()

    response = answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
        graph_store=FakeGraphStore(raises=True),
        graph_retrieval_enabled=True,
    )

    assert response["sources"][0]["source_type"] == "pdf"
    assert "Knowledge Graph Context" not in llm.calls[-1]["user_prompt"]


def test_answer_with_rag_preserves_source_scope_for_graph() -> None:
    graph_store = FakeGraphStore()

    answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        source_types=["pdf"],
        pdf_ids=["pdf-1"],
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        graph_store=graph_store,
        graph_retrieval_enabled=True,
    )

    assert graph_store.calls[0][1]["source_types"] == ["pdf"]
    assert graph_store.calls[0][1]["source_ids"] == ["pdf-1"]


def test_answer_with_rag_returns_knowledge_graph_source_when_used() -> None:
    response = answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [_result()],
        graph_store=FakeGraphStore(),
        graph_retrieval_enabled=True,
    )

    assert any(source["source_type"] == "knowledge_graph" for source in response["sources"])
