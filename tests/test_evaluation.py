"""
Tests for the evaluation harness loop.

The pipeline and the judge are both stubbed, so this checks the harness's own
arithmetic rather than the quality of any model. That arithmetic is what every
published number in this project rests on, and a mistake in it would not raise -
it would just report a different score.
"""

import json

import pytest

import evaluation
from evaluation import EvaluationResult, run_evaluation
from rag_pipeline import Generation


TWO_CASES = [
    {
        "query": "What is backpropagation used for?",
        "expected_source": "Backpropagation",
        "reference_answer": "Computing gradients via the chain rule.",
    },
    {
        "query": "What is overfitting?",
        "expected_source": "Overfitting",
        "reference_answer": "Fitting training noise and generalising poorly.",
    },
]


@pytest.fixture
def stub_harness(monkeypatch):
    """
    Drive the loop with scripted retrieval and grading outcomes.

    Returns a function taking two lists of booleans - which questions retrieve
    the right source, and which are graded correct - so each test states the
    scenario it is checking in one line.
    """

    def configure(retrieval_ok, grading_ok, latencies=None, tokens=None, parsed=None):
        latencies = latencies or [1.0] * len(retrieval_ok)
        tokens = tokens or [100] * len(retrieval_ok)
        parsed = [True] * len(retrieval_ok) if parsed is None else parsed
        state = {"i": 0}

        def fake_answer_question(query, top_k=4, verbose=False, client=None):
            i = state["i"]
            source = TWO_CASES[i]["expected_source"] if retrieval_ok[i] else "Some other article"
            return {
                "query": query,
                "retrieved_chunks": [{"source_title": source, "chunk_text": "...", "distance": 0.1}],
                "answer": "an answer",
                "generation": Generation(
                    answer="an answer",
                    prompt_tokens=tokens[i] - 20,
                    completion_tokens=20,
                    total_tokens=tokens[i],
                    latency_s=latencies[i],
                ),
            }

        def fake_grade(query, reference_answer, model_answer, client=None):
            i = state["i"]
            state["i"] += 1
            return {
                "correct": grading_ok[i],
                "reasoning": "because",
                "parsed": parsed[i],
                "raw_verdict": "VERDICT: CORRECT",
            }

        monkeypatch.setattr(evaluation, "answer_question", fake_answer_question)
        monkeypatch.setattr(evaluation, "grade_answer_with_llm", fake_grade)

    return configure


def test_all_correct_scores_full_marks(stub_harness):
    stub_harness([True, True], [True, True])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.retrieval_accuracy == 1.0
    assert result.answer_accuracy == 1.0


def test_all_wrong_scores_zero(stub_harness):
    stub_harness([False, False], [False, False])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.retrieval_accuracy == 0.0
    assert result.answer_accuracy == 0.0


def test_retrieval_and_answer_are_scored_independently(stub_harness):
    """
    The project's headline finding is a question that retrieves the wrong source
    but is still answered correctly. The two scores must not be coupled.
    """
    stub_harness(retrieval_ok=[True, False], grading_ok=[True, True])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.retrieval_accuracy == 0.5
    assert result.answer_accuracy == 1.0


def test_means_are_averaged_over_the_suite(stub_harness):
    stub_harness([True, True], [True, True], latencies=[1.0, 3.0], tokens=[100, 300])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.mean_latency_s == 2.0
    assert result.mean_tokens_per_query == 200.0


def test_top_k_is_recorded_on_the_result(stub_harness):
    stub_harness([True, True], [True, True])

    result = run_evaluation(top_k=8, test_cases=TWO_CASES, verbose=False)

    assert result.top_k == 8


def test_unreadable_verdicts_are_counted(stub_harness):
    """
    An unparseable judge reply scores INCORRECT, but it is counted separately so
    a run degraded by a flaky judge does not look like a run that genuinely
    scored badly.
    """
    stub_harness([True, True], [False, True], parsed=[False, True])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.unparsed_verdicts == 1
    assert result.answer_accuracy == 0.5


def test_per_case_rows_carry_cost(stub_harness):
    stub_harness([True, True], [True, True], latencies=[1.5, 2.5], tokens=[111, 222])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert [r["total_tokens"] for r in result.results] == [111, 222]
    assert [r["latency_s"] for r in result.results] == [1.5, 2.5]


def test_result_serialises_with_derived_accuracies(stub_harness, tmp_path):
    """The saved JSON must contain the accuracies, not just the raw hit counts."""
    stub_harness([True, False], [True, True])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)
    path = tmp_path / "results.json"
    result.save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["retrieval_accuracy"] == 0.5
    assert saved["answer_accuracy"] == 1.0
    assert saved["grading_method"] == "llm_judge"
    assert len(saved["results"]) == 2


def test_accuracies_are_derived_not_stored():
    """
    Guards against the hit counts and the accuracies drifting apart: the
    accuracy is always computed from the counts, never set independently.
    """
    result = EvaluationResult(
        top_k=4,
        retrieval_hits=9,
        answer_hits=10,
        answerable_total=10,
        mean_latency_s=1.0,
        mean_tokens_per_query=100.0,
        unparsed_verdicts=0,
    )

    assert result.retrieval_accuracy == 0.9
    assert result.answer_accuracy == 1.0


def test_the_real_benchmark_suite_is_well_formed():
    """Every case needs all three fields, or scoring silently misbehaves."""
    assert len(evaluation.TEST_CASES) == 10

    for case in evaluation.TEST_CASES:
        assert case["query"]
        assert case["expected_source"]
        assert case["reference_answer"]


# --------------------------------------------------------------------------
# Rank-aware retrieval scoring
# --------------------------------------------------------------------------


def make_result_with_ranks(ranks, refusal_hits=0, refusal_total=0):
    rows = [{"retrieval_rank": r, "unanswerable": False} for r in ranks]
    rows += [{"retrieval_rank": None, "unanswerable": True} for _ in range(refusal_total)]
    return EvaluationResult(
        top_k=4,
        retrieval_hits=sum(1 for r in ranks if r),
        answer_hits=len(ranks),
        answerable_total=len(ranks),
        refusal_hits=refusal_hits,
        refusal_total=refusal_total,
        mean_latency_s=1.0,
        mean_tokens_per_query=100.0,
        unparsed_verdicts=0,
        results=rows,
    )


@pytest.mark.parametrize(
    "ranks, expected_mrr",
    [
        ([1], 1.0),
        ([2], 0.5),
        ([4], 0.25),
        ([None], 0.0),
        ([1, 1, 2, 2], 0.75),
        ([1, None], 0.5),
    ],
)
def test_mrr_weights_by_rank(ranks, expected_mrr):
    """Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25, a miss scores 0."""
    assert make_result_with_ranks(ranks).mean_reciprocal_rank == pytest.approx(expected_mrr)


def test_mrr_distinguishes_what_the_hit_rate_cannot():
    """
    The whole point of adding it. Two pipelines that both put the right article
    in the window score identically on hit rate; only one of them ranks it first.
    """
    ranks_first = make_result_with_ranks([1, 1, 1, 1])
    ranks_later = make_result_with_ranks([2, 2, 2, 2])

    assert ranks_first.retrieval_accuracy == ranks_later.retrieval_accuracy == 1.0
    assert ranks_first.mean_reciprocal_rank == 1.0
    assert ranks_later.mean_reciprocal_rank == 0.5


def test_rank_1_rate_counts_only_first_place():
    assert make_result_with_ranks([1, 2, 1, None]).rank_1_rate == 0.5


def test_mrr_on_the_real_sweep_data():
    """
    Reproduces the figure computed from the logged chunk250 runs: six questions
    at rank 1 and four at rank 2 gives 0.800. If the implementation disagrees
    with the recorded artifacts, one of them is wrong.
    """
    ranks = [1] * 6 + [2] * 4
    assert make_result_with_ranks(ranks).mean_reciprocal_rank == pytest.approx(0.800)


def test_rank_is_recorded_per_question(stub_harness):
    stub_harness([True, False], [True, True])

    result = run_evaluation(test_cases=TWO_CASES, verbose=False)

    assert result.results[0]["retrieval_rank"] == 1
    assert result.results[1]["retrieval_rank"] is None


# --------------------------------------------------------------------------
# Refusal questions
# --------------------------------------------------------------------------


UNANSWERABLE = [{"query": "What is the capital city of Australia?", "unanswerable": True}]


@pytest.fixture
def stub_refusal(monkeypatch):
    """Drive the loop with a scripted answer and refusal verdict."""

    def configure(answer, refused):
        def fake_answer_question(query, top_k=4, verbose=False, client=None):
            return {
                "query": query,
                "retrieved_chunks": [{"source_title": "Overfitting", "chunk_text": "...", "distance": 0.9}],
                "answer": answer,
                "generation": Generation(answer, 90, 10, 100, 1.0),
            }

        def fake_grade_refusal(query, model_answer, client=None):
            return {"correct": refused, "reasoning": "because", "parsed": True, "raw_verdict": ""}

        monkeypatch.setattr(evaluation, "answer_question", fake_answer_question)
        monkeypatch.setattr(evaluation, "grade_refusal", fake_grade_refusal)

    return configure


def test_refusal_is_scored_separately_from_answer_accuracy(stub_refusal):
    """
    Averaging "declined correctly" into the same figure as "answered correctly"
    would produce a number meaning neither.
    """
    stub_refusal("I don't have enough information to answer that.", refused=True)

    result = run_evaluation(test_cases=UNANSWERABLE, verbose=False)

    assert result.refusal_accuracy == 1.0
    assert result.refusal_total == 1
    assert result.answerable_total == 0


def test_answering_an_unanswerable_question_is_a_refusal_failure(stub_refusal):
    """The classic RAG failure: confidently answering from pretraining."""
    stub_refusal("The capital of Australia is Canberra.", refused=False)

    result = run_evaluation(test_cases=UNANSWERABLE, verbose=False)

    assert result.refusal_accuracy == 0.0


def test_unanswerable_questions_leave_the_retrieval_denominator_alone(stub_refusal):
    """
    They have no correct source. Counting one as a miss would understate
    retrieval; counting it as a hit would inflate it. It must not count at all.
    """
    stub_refusal("I don't have enough information to answer that.", refused=True)

    result = run_evaluation(test_cases=UNANSWERABLE, verbose=False)

    assert result.answerable_total == 0
    assert result.retrieval_hits == 0
    assert result.total == 1


def test_refusal_accuracy_is_none_when_no_refusal_questions_ran():
    """Not 0.0 - that would claim the model never refuses, which was not tested."""
    assert make_result_with_ranks([1, 2]).refusal_accuracy is None


def test_mixed_suite_keeps_the_denominators_apart():
    result = make_result_with_ranks([1, 1], refusal_hits=3, refusal_total=5)

    assert result.answerable_total == 2
    assert result.refusal_total == 5
    assert result.total == 7
    assert result.retrieval_accuracy == 1.0
    assert result.refusal_accuracy == 0.6


def test_benchmark_suites_are_versioned_and_v1_is_frozen():
    """
    v1 is what all six sweep configurations are measured on. If it ever gains a
    question, the comparison between those runs silently stops being valid.
    """
    assert len(evaluation.BENCHMARK_V1) == 10
    assert len(evaluation.BENCHMARK_V2) == 15
    assert evaluation.BENCHMARK_V1 == evaluation.TEST_CASES
    assert all(not c.get("unanswerable") for c in evaluation.BENCHMARK_V1)
    assert sum(1 for c in evaluation.BENCHMARK_V2 if c.get("unanswerable")) == 5


def test_run_defaults_to_v1(stub_harness):
    stub_harness([True, True], [True, True])
    result = run_evaluation(test_cases=TWO_CASES, verbose=False)
    assert result.benchmark_version == evaluation.BENCHMARK_V1_NAME
