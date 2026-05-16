from rag_pipeline.pedagogical_prompts import AL_STATE_CLOSE, AL_STATE_OPEN
from rag_pipeline.rag_answer import FALLBACK_ANSWER, answer_with_rag


class FakeLlmClient:
    def __init__(self, answer: str = "Antwort") -> None:
        self.answer = answer
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.answer


def _result() -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": "Process Mining verbindet Data Science und Process Science.",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "page_index": 0,
        "title": "GPAA.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA.pdf"},
    }


def test_normal_mode_uses_normal_prompt() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", chat_mode="normal", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    prompt = llm.calls[-1]["system_prompt"].lower()
    assert "socratic tutor" not in prompt
    assert "5-year-old" not in prompt


def test_guided_learning_mode_uses_guided_prompt() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", chat_mode="guided_learning", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    assert "socratic tutor" in llm.calls[-1]["system_prompt"].lower()


def test_feynman_mode_uses_feynman_prompt() -> None:
    llm = FakeLlmClient()

    answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="feynman",
        active_learning_state={"mode": "feynman", "learner_name": "Mira", "language": "de"},
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert "5-year-old" in llm.calls[-1]["system_prompt"].lower()
    assert '"learner_name":"Mira"' in llm.calls[-1]["user_prompt"]


def test_feynman_prompt_tells_model_to_respond_to_existing_explanation() -> None:
    llm = FakeLlmClient("Follow-up")

    answer_with_rag(
        "use data to analyze processes and find out inefficiencies",
        "user-1",
        chat_mode="feynman",
        active_learning_state={"mode": "feynman", "current_step": "awaiting_explanation", "learner_name": "Lam Anh"},
        recent_messages=[
            {"role": "assistant", "content": "Hi Lam Anh, what will you explain to me today?"},
            {"role": "user", "content": "Im gonna explain process mining to you today"},
            {"role": "assistant", "content": "Can you explain what Process Mining is in very simple words?"},
        ],
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    prompt = llm.calls[-1]["system_prompt"].lower()
    assert "do not repeat the initial invitation" in prompt
    assert "respond to the learner's latest" in prompt
    assert "explanation directly" in prompt
    assert "what will you explain" not in prompt


def test_guided_learning_forwards_pdf_ids_unchanged() -> None:
    calls = []

    answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="guided_learning",
        pdf_ids=["id1"],
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **kwargs: calls.append(kwargs) or [_result()],
    )

    assert calls[0]["pdf_ids"] == ["id1"]


def test_feynman_forwards_pdf_ids_unchanged() -> None:
    calls = []

    answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="feynman",
        pdf_ids=["id1"],
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **kwargs: calls.append(kwargs) or [_result()],
    )

    assert calls[0]["pdf_ids"] == ["id1"]


def test_guided_learning_returns_updated_state() -> None:
    llm = FakeLlmClient(f"Answer{AL_STATE_OPEN}{'{\"current_step\":\"ask_question\"}'}{AL_STATE_CLOSE}")

    response = answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="guided_learning",
        active_learning_state={"mode": "guided_learning"},
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert response["answer"] == "Answer"
    assert response["updated_active_learning_state"]["current_step"] == "ask_question"


def test_feynman_returns_updated_state() -> None:
    llm = FakeLlmClient(f"Answer{AL_STATE_OPEN}{'{\"current_step\":\"evaluate_answer\"}'}{AL_STATE_CLOSE}")

    response = answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="feynman",
        active_learning_state={"mode": "feynman"},
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert response["updated_active_learning_state"]["current_step"] == "evaluate_answer"


def test_normal_mode_does_not_return_updated_state() -> None:
    response = answer_with_rag("Frage", "user-1", llm_client=FakeLlmClient(), retrieval_fn=lambda **_: [_result()])

    assert "updated_active_learning_state" not in response


def test_active_learning_state_merge_preserves_mode() -> None:
    llm = FakeLlmClient(
        f"Answer{AL_STATE_OPEN}"
        '{"mode":"feynman","current_step":"evaluate_answer"}'
        f"{AL_STATE_CLOSE}"
    )

    response = answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="guided_learning",
        active_learning_state={"mode": "guided_learning", "current_step": "ask_question"},
        llm_client=llm,
        retrieval_fn=lambda **_: [_result()],
    )

    assert response["updated_active_learning_state"]["mode"] == "guided_learning"
    assert response["updated_active_learning_state"]["current_step"] == "evaluate_answer"


def test_fallback_answer_unchanged_for_active_learning_without_results() -> None:
    response = answer_with_rag(
        "Frage",
        "user-1",
        chat_mode="guided_learning",
        llm_client=FakeLlmClient(),
        retrieval_fn=lambda **_: [],
    )

    assert response["answer"] == FALLBACK_ANSWER
    assert "updated_active_learning_state" not in response
