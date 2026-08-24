"""
Tests for the LLM-judge grading rubric.

The evaluation harness is only trustworthy if the judge's verdict is read
correctly. A parsing bug here does not raise - it silently moves the headline
accuracy number, which is exactly the failure mode these tests exist to catch.
"""

import pytest

from conftest import FakeGroqClient
from config import REFUSAL_MARKER
from grading import (
    build_grading_prompt,
    build_refusal_prompt,
    grade_answer_with_llm,
    grade_refusal,
    is_refusal,
    parse_verdict,
)


# --------------------------------------------------------------------------
# parse_verdict - pure string handling, no mocking needed
# --------------------------------------------------------------------------


def test_plain_correct_verdict():
    result = parse_verdict("VERDICT: CORRECT\nREASONING: covers the mechanism.")
    assert result["correct"] is True
    assert result["parsed"] is True
    assert result["reasoning"] == "covers the mechanism."


def test_plain_incorrect_verdict():
    result = parse_verdict("VERDICT: INCORRECT\nREASONING: bare category name, rule 2.")
    assert result["correct"] is False
    assert result["parsed"] is True
    assert result["reasoning"] == "bare category name, rule 2."


def test_incorrect_is_not_read_as_correct():
    """
    'INCORRECT' contains 'CORRECT'. A naive substring check gets this backwards
    and scores every failing answer as a pass, so it is pinned explicitly.
    """
    assert parse_verdict("VERDICT: INCORRECT")["correct"] is False


@pytest.mark.parametrize(
    "reply",
    [
        "VERDICT: [CORRECT]",           # judge echoes the prompt's own template
        "**VERDICT: CORRECT**",         # judge bolds its headings
        "verdict: correct",             # lower case
        "  VERDICT:   CORRECT  ",       # ragged whitespace
        "VERDICT: (CORRECT)",           # parenthesised
        "Here is my grade.\nVERDICT: CORRECT\nREASONING: fine.",  # preamble
    ],
)
def test_correct_verdict_survives_formatting_noise(reply):
    """
    The grading prompt shows the judge the literal template
    "VERDICT: [CORRECT or INCORRECT]", so echoed brackets are not hypothetical.
    The original substring check failed on that case and recorded a silent
    INCORRECT.
    """
    result = parse_verdict(reply)
    assert result["correct"] is True
    assert result["parsed"] is True


@pytest.mark.parametrize(
    "reply",
    ["VERDICT: [INCORRECT]", "**VERDICT: INCORRECT**", "verdict: incorrect"],
)
def test_incorrect_verdict_survives_formatting_noise(reply):
    result = parse_verdict(reply)
    assert result["correct"] is False
    assert result["parsed"] is True


def test_unreadable_reply_is_flagged_not_silently_failed():
    """
    An unparseable reply still scores INCORRECT - rule 5 of the rubric says to
    resolve doubt that way - but `parsed` distinguishes it from a real
    INCORRECT so the harness can warn instead of quietly losing a point.
    """
    result = parse_verdict("I think the answer is pretty good overall.")
    assert result["correct"] is False
    assert result["parsed"] is False


def test_reasoning_is_optional():
    result = parse_verdict("VERDICT: CORRECT")
    assert result["correct"] is True
    assert result["reasoning"] == ""


def test_reasoning_containing_a_colon_is_not_truncated():
    result = parse_verdict(
        "VERDICT: INCORRECT\nREASONING: rule 2: names the category without a mechanism."
    )
    assert result["reasoning"] == "rule 2: names the category without a mechanism."


def test_verdict_must_start_a_line():
    """
    Prose that merely mentions the phrase should not be mistaken for the
    verdict line itself.
    """
    assert parse_verdict("I would not say VERDICT: CORRECT here.")["parsed"] is False


# --------------------------------------------------------------------------
# The rubric prompt
# --------------------------------------------------------------------------


def test_grading_prompt_carries_all_three_inputs():
    prompt = build_grading_prompt(
        query="What is overfitting?",
        reference_answer="Fitting training noise and generalising poorly.",
        model_answer="It memorises the training set.",
    )

    assert "What is overfitting?" in prompt
    assert "Fitting training noise and generalising poorly." in prompt
    assert "It memorises the training set." in prompt


def test_grading_prompt_keeps_the_ordered_rubric():
    """
    The ordered rules are what made grading consistent between runs; the
    bare-category rule in particular is what the recorded results depend on.
    Losing them in an edit would change the measurements silently.
    """
    prompt = build_grading_prompt("q", "ref", "ans")

    for rule_number in range(1, 6):
        assert f"{rule_number}." in prompt
    assert "must be marked INCORRECT" in prompt
    assert "Do not give partial credit" in prompt
    assert "VERDICT:" in prompt


# --------------------------------------------------------------------------
# grade_answer_with_llm - the judge client is injected, so no key is needed
# --------------------------------------------------------------------------


def test_grade_answer_uses_the_injected_client_and_returns_the_raw_reply():
    client = FakeGroqClient("VERDICT: CORRECT\nREASONING: explains the mechanism.")

    result = grade_answer_with_llm("q", "ref", "ans", client=client)

    assert result["correct"] is True
    assert result["reasoning"] == "explains the mechanism."
    assert result["raw_verdict"] == "VERDICT: CORRECT\nREASONING: explains the mechanism."


def test_grade_answer_grades_deterministically():
    """Temperature 0 - a judge that varies between runs is not a measurement."""
    client = FakeGroqClient("VERDICT: CORRECT")

    grade_answer_with_llm("q", "ref", "ans", client=client)

    assert client.calls[0]["temperature"] == 0.0


def test_judge_is_a_different_model_from_the_generator():
    """
    A model grading its own output has a self-preference bias, so the judge is
    deliberately independent of the generator. If these two config values ever
    collapse to the same string, the harness quietly stops being independent.
    """
    from config import GROQ_MODEL, JUDGE_MODEL

    client = FakeGroqClient("VERDICT: CORRECT")
    grade_answer_with_llm("q", "ref", "ans", client=client)

    assert client.calls[0]["model"] == JUDGE_MODEL
    assert JUDGE_MODEL != GROQ_MODEL


def test_grade_answer_sends_the_rubric_prompt():
    client = FakeGroqClient("VERDICT: CORRECT")

    grade_answer_with_llm("What is overfitting?", "ref text", "ans text", client=client)

    sent = client.calls[0]["messages"][0]["content"]
    assert sent == build_grading_prompt("What is overfitting?", "ref text", "ans text")


# --------------------------------------------------------------------------
# Refusal detection and grading
# --------------------------------------------------------------------------


def test_canonical_refusal_is_recognised():
    assert is_refusal(REFUSAL_MARKER) is True


@pytest.mark.parametrize(
    "reply",
    [
        "I don't have enough information to answer that.",
        "I don\u2019t have enough information to answer that.",   # curly apostrophe
        "  i don't have enough information to answer that  ",     # case and whitespace
        "**I don't have enough information to answer that.**",    # bolded
        "I don't have enough information to answer that. The context covers ML topics.",
    ],
)
def test_canonical_refusal_survives_formatting_noise(reply):
    """
    The model reproduces the requested wording but rarely byte-for-byte. A
    literal equality check would spend a judge call on every one of these.
    """
    assert is_refusal(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        "The capital of Australia is Canberra.",
        "Backpropagation computes gradients using the chain rule.",
        "",
    ],
)
def test_substantive_answers_are_not_refusals(reply):
    assert is_refusal(reply) is False


def test_canonical_refusal_costs_no_judge_call():
    """
    The cost argument for this design: when the model complies, grading a
    refusal is free. A judge call is spent only on paraphrases.
    """
    client = FakeGroqClient("VERDICT: DECLINED")

    result = grade_refusal("What is the capital of Australia?", REFUSAL_MARKER, client=client)

    assert result["correct"] is True
    assert result["parsed"] is True
    assert client.calls == [], "a canonical refusal must not reach the judge"


def test_paraphrased_refusal_falls_back_to_the_judge():
    client = FakeGroqClient("VERDICT: DECLINED\nREASONING: it declined, reworded.")

    result = grade_refusal(
        "What is the capital of Australia?",
        "The provided context does not cover geography, so I cannot answer.",
        client=client,
    )

    assert result["correct"] is True
    assert len(client.calls) == 1


def test_judge_says_answered_when_the_model_did_not_decline():
    client = FakeGroqClient("VERDICT: ANSWERED\nREASONING: gave a substantive answer.")

    result = grade_refusal(
        "What is the capital of Australia?",
        "The capital of Australia is Canberra.",
        client=client,
    )

    assert result["correct"] is False
    assert result["parsed"] is True


def test_answered_is_not_read_as_declined():
    """DECLINED and ANSWERED share no substring, but pin the alternation anyway."""
    client = FakeGroqClient("VERDICT: [ANSWERED]")
    assert grade_refusal("q", "some answer", client=client)["correct"] is False


def test_unreadable_refusal_verdict_is_flagged():
    client = FakeGroqClient("I am not sure what you are asking me to do.")

    result = grade_refusal("q", "some answer", client=client)

    assert result["correct"] is False
    assert result["parsed"] is False


def test_refusal_rubric_judges_declining_not_correctness():
    """
    Rule 2 is the load-bearing one. A confidently correct answer drawn from
    pretraining is precisely the failure being measured, so the judge must not
    reward it for being factually right.
    """
    prompt = build_refusal_prompt("What is the capital of Australia?", "Canberra.")

    assert "DECLINED" in prompt
    assert "ANSWERED" in prompt
    assert "even if" in prompt and "factually correct" in prompt
    assert "Correctness is not what is being judged here." in prompt


def test_refusal_grading_returns_the_same_shape_as_answer_grading():
    """One result contract, so the harness does not branch on question type."""
    client = FakeGroqClient("VERDICT: CORRECT\nREASONING: fine.")
    answer_keys = set(grade_answer_with_llm("q", "ref", "ans", client=client))

    canonical = set(grade_refusal("q", REFUSAL_MARKER))
    judged = set(grade_refusal("q", "a paraphrase", client=FakeGroqClient("VERDICT: DECLINED")))

    assert canonical == answer_keys
    assert judged == answer_keys


def test_refusal_marker_matches_the_generation_prompt():
    """
    The two must never drift. If the prompt asks for one string and the grader
    checks another, every refusal would be scored as a failure.
    """
    from rag_pipeline import build_prompt

    assert REFUSAL_MARKER in build_prompt("q", [])
