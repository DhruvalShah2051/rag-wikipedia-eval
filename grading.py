"""
LLM-as-judge grading for the evaluation harness.

Extracted from 5_evaluate.py so the rubric and, more importantly, the verdict
parsing can be imported and tested. 5_evaluate.py keeps the benchmark suite and
the run loop; this module owns the question "is this answer correct?".

The grading prompt below is reproduced verbatim from the version that produced
the recorded evaluation results. Changing its wording changes the measurements,
so treat it as a fixed artifact and re-run the harness if it is ever edited.
"""

import re

from groq import Groq
from config import GROQ_API_KEY, JUDGE_MODEL
from llm import chat_completion

# Bumped whenever the rubric text below changes. Logged to MLflow as a run
# parameter: two runs graded by different rubrics are not comparable, and
# without a version recorded alongside the score there is no way to tell.
# v3 is the strict ordered rubric described in the README's Results section -
# the one every recorded number was produced under.
JUDGE_PROMPT_VERSION = "v3-strict-rubric"

# Constructed on first use rather than at import time, so this module can be
# imported without a GROQ_API_KEY present (tests, CI, tooling).
_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = Groq(api_key=GROQ_API_KEY)
    return _judge_client


def build_grading_prompt(query, reference_answer, model_answer):
    """
    Assemble the judge prompt: an explicit, ordered rubric rather than a vague
    "is this good?". The ordered rules are what made grading consistent between
    runs - an earlier, looser prompt graded structurally identical answers
    differently on different passes.
    """
    return f"""You are grading the correctness of an AI-generated answer. Apply these rules consistently and strictly.

QUESTION: {query}

REFERENCE ANSWER (the key facts a correct answer should convey):
{reference_answer}

MODEL'S ANSWER:
{model_answer}

GRADING RULES (apply these in order, and apply them the same way every time):
1. The wording does not need to match the reference answer - judge based on factual correctness only.
2. A bare category name or one-word/one-phrase answer with NO supporting mechanism, reasoning, or explanation
   must be marked INCORRECT, even if that category name is technically the right answer. A correct answer
   must explain the "how" or "why", not just name the "what".
3. Minor omissions of secondary/tertiary detail are acceptable IF the core mechanism or reasoning is present
   and correct.
4. If the answer contradicts or omits the core mechanism described in the reference answer, mark it INCORRECT.
5. Do not give partial credit - the verdict is binary. If in doubt between CORRECT and INCORRECT, choose
   INCORRECT and explain the specific missing element.

Respond in this exact format:
VERDICT: [CORRECT or INCORRECT]
REASONING: [one sentence explaining why, citing the specific rule applied]"""


# The judge is shown the template "VERDICT: [CORRECT or INCORRECT]", so it
# sometimes echoes the square brackets - and small models like to bold their
# headings. Both patterns tolerate that surrounding noise.
#
# INCORRECT is listed first in the alternation deliberately: it contains
# "CORRECT" as a substring, so a CORRECT-first pattern would mis-grade every
# failing answer as a pass.
_VERDICT_RE = re.compile(
    r"^[\s*_#]*VERDICT[\s*_]*:[\s*_\[\(]*(INCORRECT|CORRECT)\b",
    re.IGNORECASE | re.MULTILINE,
)

_REASONING_RE = re.compile(
    r"^[\s*_#]*REASONING[\s*_]*:[\s*_\[\(]*(.+?)[\s*_\]\)]*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdict(result_text):
    """
    Pull the verdict and reasoning out of the judge's reply.

    Returns a dict with:
      correct   - True only on an explicit CORRECT verdict
      reasoning - the judge's one-line justification, or "" if absent
      parsed    - False if no verdict line was found at all

    When no verdict can be read, `correct` is False. That follows the rubric's
    own rule 5 (when in doubt, INCORRECT), but `parsed` is there so the caller
    can tell a real INCORRECT apart from an unreadable reply instead of
    silently scoring it as a failure.
    """
    verdict_match = _VERDICT_RE.search(result_text)
    reasoning_match = _REASONING_RE.search(result_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if verdict_match is None:
        return {"correct": False, "reasoning": reasoning, "parsed": False}

    return {
        "correct": verdict_match.group(1).upper() == "CORRECT",
        "reasoning": reasoning,
        "parsed": True,
    }


def grade_answer_with_llm(query, reference_answer, model_answer, client=None):
    """
    Use Groq as an LLM judge to grade whether the model's answer correctly
    addresses the question, compared against a short reference answer.

    `client` is injectable so tests can grade against a stub without a network
    call or an API key.
    """
    response, _latency_s = chat_completion(
        client or _get_judge_client(),
        model=JUDGE_MODEL,
        prompt=build_grading_prompt(query, reference_answer, model_answer),
        temperature=0.0,  # deterministic grading
    )

    result_text = response.choices[0].message.content.strip()

    result = parse_verdict(result_text)
    result["raw_verdict"] = result_text
    return result
