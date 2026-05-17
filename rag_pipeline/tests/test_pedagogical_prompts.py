from rag_pipeline.pedagogical_prompts import (
    AL_STATE_CLOSE,
    AL_STATE_OPEN,
    FEYNMAN_RESULT_SYSTEM_PROMPT,
    FEYNMAN_SYSTEM_PROMPT,
    GUIDED_LEARNING_SYSTEM_PROMPT,
    extract_al_state_update,
)


def test_guided_learning_prompt_mentions_question_and_tutor():
    assert isinstance(GUIDED_LEARNING_SYSTEM_PROMPT, str)
    assert GUIDED_LEARNING_SYSTEM_PROMPT.strip()
    prompt = GUIDED_LEARNING_SYSTEM_PROMPT.lower()
    assert "question" in prompt
    assert "tutor" in prompt
    assert "ask one focused question at a time" in prompt
    assert "give a small hint, not the full solution" in prompt
    assert "do not dump a full lesson" in prompt
    assert "do not ask multiple questions at once" in prompt


def test_feynman_prompt_mentions_beginner_explanation():
    assert isinstance(FEYNMAN_SYSTEM_PROMPT, str)
    assert FEYNMAN_SYSTEM_PROMPT.strip()
    prompt = FEYNMAN_SYSTEM_PROMPT.lower()
    assert "curious beginner" in prompt
    assert "learner_name" in prompt
    assert "not pretend to literally be a child" in prompt
    assert "ask only one main question at a time" in prompt
    assert "do not respond with only a question" in prompt
    assert "do not validate like a teacher" in prompt


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


def test_extract_al_state_update_whitelists_keys_and_clamps_score():
    answer = (
        "Keep going.\n"
        f"{AL_STATE_OPEN}"
        '{"current_step":"ask_question","user_understanding_score":1.7,'
        '"unexpected_key":"ignored","mode":"feynman"}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "guided_learning", "topic": "Plants"}

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["mode"] == "guided_learning"
    assert merged_state["topic"] == "Plants"
    assert merged_state["current_step"] == "ask_question"
    assert merged_state["user_understanding_score"] == 1.0
    assert "unexpected_key" not in merged_state


def test_extract_al_state_update_sanitizes_state_values():
    answer = (
        "Keep going.\n"
        f"{AL_STATE_OPEN}"
        '{"current_step":"  ask_question  ","language":"fr",'
        '"target_concept":"  Process Mining  ",'
        '"user_understanding_score":true,'
        '"covered_concepts":[" Event Logs ","Event Logs",42,""],'
        '"asked_questions":["Q1","Q2"],'
        '"last_question":"  What does that mean here?  "}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {
        "mode": "feynman",
        "language": "de",
        "user_understanding_score": 0.25,
    }

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["mode"] == "feynman"
    assert merged_state["language"] == "de"
    assert merged_state["user_understanding_score"] == 0.25
    assert merged_state["current_step"] == "ask_question"
    assert merged_state["target_concept"] == "Process Mining"
    assert merged_state["covered_concepts"] == ["Event Logs"]
    assert merged_state["asked_questions"] == ["Q1", "Q2"]
    assert merged_state["last_question"] == "What does that mean here?"


def test_extract_al_state_update_limits_list_values():
    concepts = [f"concept-{index}" for index in range(25)]
    answer = (
        "Keep going.\n"
        f"{AL_STATE_OPEN}"
        + '{"covered_concepts":'
        + str(concepts).replace("'", '"')
        + "}"
        + f"{AL_STATE_CLOSE}"
    )

    _, merged_state = extract_al_state_update(answer, {})

    assert len(merged_state["covered_concepts"]) == 20
    assert merged_state["covered_concepts"][-1] == "concept-19"


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


def test_feynman_prompt_contains_completion_behavior():
    prompt = FEYNMAN_SYSTEM_PROMPT.lower()
    assert "completion behavior" in prompt
    assert "completion criteria" in prompt
    assert "ready_for_result" in prompt
    assert "final_check" in prompt
    assert "/fertig" in prompt
    assert "should_nudge_completion" in prompt


def test_feynman_result_prompt_structure():
    assert isinstance(FEYNMAN_RESULT_SYSTEM_PROMPT, str)
    prompt = FEYNMAN_RESULT_SYSTEM_PROMPT
    assert "Kurzfazit" in prompt
    assert "Was du schon" in prompt
    assert "Missverst" in prompt
    assert "Verbesserte Mini-Erkl" in prompt
    assert "exercise_status" in prompt
    assert "completed" in prompt
    lower = prompt.lower()
    assert "turn_count" in lower
    assert "mode" in lower


def test_extract_al_state_update_accepts_new_completion_keys():
    answer = (
        "Almost there."
        f"{AL_STATE_OPEN}"
        '{"exercise_status":"final_check","completion_readiness":0.78,'
        '"remaining_gaps":[" event logs "," ","duplicate","duplicate"],'
        '"final_check_question":"  Was zeigt die Karte wirklich?  "}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "feynman"}

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["exercise_status"] == "final_check"
    assert merged_state["completion_readiness"] == 0.78
    assert merged_state["remaining_gaps"] == ["event logs", "duplicate"]
    assert merged_state["final_check_question"] == "Was zeigt die Karte wirklich?"
    assert merged_state["mode"] == "feynman"


def test_extract_al_state_update_rejects_invalid_exercise_status():
    answer = (
        "Progress."
        f"{AL_STATE_OPEN}"
        '{"exercise_status":"some-other-status"}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "feynman", "exercise_status": "active"}

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["exercise_status"] == "active"


def test_extract_al_state_update_clamps_completion_readiness():
    answer = (
        "Progress."
        f"{AL_STATE_OPEN}"
        '{"completion_readiness":1.4}'
        f"{AL_STATE_CLOSE}"
    )
    _, merged_state = extract_al_state_update(answer, {"mode": "feynman"})
    assert merged_state["completion_readiness"] == 1.0

    answer_neg = (
        "Progress."
        f"{AL_STATE_OPEN}"
        '{"completion_readiness":-0.3}'
        f"{AL_STATE_CLOSE}"
    )
    _, merged_neg = extract_al_state_update(answer_neg, {"mode": "feynman"})
    assert merged_neg["completion_readiness"] == 0.0

    answer_bad = (
        "Progress."
        f"{AL_STATE_OPEN}"
        '{"completion_readiness":"high"}'
        f"{AL_STATE_CLOSE}"
    )
    _, merged_bad = extract_al_state_update(answer_bad, {"mode": "feynman"})
    assert "completion_readiness" not in merged_bad


def test_extract_al_state_update_preserves_server_turn_count():
    answer = (
        "Almost there."
        f"{AL_STATE_OPEN}"
        '{"turn_count":99,"current_step":"ask_question"}'
        f"{AL_STATE_CLOSE}"
    )
    current_state = {"mode": "feynman", "turn_count": 4}

    _, merged_state = extract_al_state_update(answer, current_state)

    assert merged_state["turn_count"] == 4
    assert merged_state["current_step"] == "ask_question"


def test_extract_al_state_update_remaining_gaps_must_be_list():
    answer = (
        "."
        f"{AL_STATE_OPEN}"
        '{"remaining_gaps":"single string"}'
        f"{AL_STATE_CLOSE}"
    )
    _, merged_state = extract_al_state_update(answer, {"mode": "feynman"})
    assert "remaining_gaps" not in merged_state
