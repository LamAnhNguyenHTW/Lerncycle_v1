from rag_pipeline.pedagogical_prompts import (
    AL_STATE_CLOSE,
    AL_STATE_OPEN,
    FEYNMAN_SYSTEM_PROMPT,
    GUIDED_LEARNING_SYSTEM_PROMPT,
    extract_al_state_update,
)


def test_guided_learning_prompt_mentions_question_and_tutor():
    assert isinstance(GUIDED_LEARNING_SYSTEM_PROMPT, str)
    assert GUIDED_LEARNING_SYSTEM_PROMPT.strip()
    assert "question" in GUIDED_LEARNING_SYSTEM_PROMPT.lower()
    assert "tutor" in GUIDED_LEARNING_SYSTEM_PROMPT.lower()


def test_feynman_prompt_mentions_child_explanation():
    assert isinstance(FEYNMAN_SYSTEM_PROMPT, str)
    assert FEYNMAN_SYSTEM_PROMPT.strip()
    prompt = FEYNMAN_SYSTEM_PROMPT.lower()
    assert "5-year-old" in prompt or "5 jahre" in prompt
    assert "learner_name" in prompt
    assert "hallo <name>" in prompt


def test_extract_al_state_update_strips_sentinel_and_merges_state():
    answer = (
        "Good start. What happens next?\n"
        f"{AL_STATE_OPEN}"
        '{"current_step": "ask_question", "covered_concepts": ["photosynthesis"]}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "guided_learning", "topic": "Plants"}

    clean_answer, merged_state = extract_al_state_update(answer, current_state)

    assert clean_answer == "Good start. What happens next?"
    assert merged_state == {
        "mode": "guided_learning",
        "topic": "Plants",
        "current_step": "ask_question",
        "covered_concepts": ["photosynthesis"],
    }


def test_extract_al_state_update_without_sentinel_returns_unchanged_values():
    answer = "No state update here."
    current_state = {"mode": "feynman", "current_step": "ask_question"}

    clean_answer, merged_state = extract_al_state_update(answer, current_state)

    assert clean_answer == answer
    assert merged_state == current_state


def test_extract_al_state_update_malformed_json_strips_sentinel_and_keeps_state():
    answer = f"Keep going.\n{AL_STATE_OPEN}not-json{AL_STATE_CLOSE}"
    current_state = {"mode": "guided_learning", "current_step": "ask_question"}

    clean_answer, merged_state = extract_al_state_update(answer, current_state)

    assert clean_answer == "Keep going."
    assert merged_state == current_state


def test_extract_al_state_update_preserves_server_authoritative_mode():
    answer = (
        "Let's continue."
        f"{AL_STATE_OPEN}"
        '{"mode": "feynman", "current_step": "evaluate_answer"}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "guided_learning", "current_step": "ask_question"}

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["mode"] == "guided_learning"
    assert merged_state["current_step"] == "evaluate_answer"


def test_extract_al_state_update_ignores_json_literals_and_citation_markers():
    answer = 'Use the formula {"a": 1} from source [1].'
    current_state = {"mode": "feynman"}

    clean_answer, merged_state = extract_al_state_update(answer, current_state)

    assert clean_answer == answer
    assert merged_state == current_state
