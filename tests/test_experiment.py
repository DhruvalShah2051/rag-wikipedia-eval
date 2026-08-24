"""
Tests for the MLflow payload builders.

No tracking server, no database, no API key - these check that a run is
described completely and correctly before anything is sent anywhere.

That matters more here than it might elsewhere. This project has twice published
a number that was measuring something other than its name: once the judge's
rubric version, once the vector index's recall. In both cases the score itself
was fine and the missing piece was the configuration recorded next to it. A
params dict that silently omits or misstates a field reintroduces exactly that
failure, and it would not raise.
"""

import pytest

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    IVFFLAT_PROBES,
    JUDGE_MODEL,
)
from evaluation import BENCHMARK_V1_NAME, EvaluationResult
from experiment import build_metrics, build_params
from grading import JUDGE_PROMPT_VERSION


def make_result(retrieval_hits=10, answer_hits=9, answerable_total=10, unparsed=0,
                ranks=None, refusal_hits=0, refusal_total=0):
    """
    An EvaluationResult with optional per-question ranks.

    MRR is derived from the result rows, so a test that cares about it has to
    supply them; `ranks` builds the minimal rows the property reads.
    """
    rows = [{"retrieval_rank": rank, "unanswerable": False} for rank in (ranks or [])]
    rows += [{"retrieval_rank": None, "unanswerable": True} for _ in range(refusal_total)]

    return EvaluationResult(
        top_k=4,
        retrieval_hits=retrieval_hits,
        answer_hits=answer_hits,
        answerable_total=answerable_total,
        refusal_hits=refusal_hits,
        refusal_total=refusal_total,
        mean_latency_s=5.5,
        mean_tokens_per_query=1600.0,
        unparsed_verdicts=unparsed,
        results=rows,
    )


# --------------------------------------------------------------------------
# build_params
# --------------------------------------------------------------------------


def test_params_cover_everything_that_makes_runs_comparable():
    params = build_params(top_k=4)

    assert set(params) == {
        "top_k",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "generation_model",
        "judge_model",
        "judge_prompt_version",
        "benchmark_version",
        "ivfflat_lists",
        "ivfflat_probes",
    }


def test_benchmark_version_is_recorded():
    """
    Runs scored on different question sets are not comparable at all. Without
    this recorded, a table mixing a 10-question suite with a 15-question one
    would look perfectly well-formed and mean nothing.
    """
    assert build_params(top_k=4)["benchmark_version"] == BENCHMARK_V1_NAME
    assert build_params(top_k=4, benchmark_version="v2-15q-refusals")[
        "benchmark_version"
    ] == "v2-15q-refusals"


def test_params_default_to_the_configured_pipeline():
    params = build_params(top_k=4)

    assert params["chunk_size"] == CHUNK_SIZE_WORDS
    assert params["chunk_overlap"] == CHUNK_OVERLAP_WORDS
    assert params["embedding_model"] == EMBEDDING_MODEL_NAME
    assert params["generation_model"] == GROQ_MODEL
    assert params["judge_model"] == JUDGE_MODEL
    assert params["ivfflat_probes"] == IVFFLAT_PROBES


def test_swept_values_override_the_defaults():
    """
    The sweep varies these per run. If they fell back to config, every swept run
    would be labelled with the baseline's configuration.
    """
    params = build_params(top_k=8, chunk_size=500, chunk_overlap=100)

    assert params["top_k"] == 8
    assert params["chunk_size"] == 500
    assert params["chunk_overlap"] == 100


def test_index_configuration_is_recorded():
    """
    The omission that let a 90% retrieval score go unquestioned. A retrieval
    number without the index settings behind it is not interpretable.
    """
    params = build_params(top_k=4, ivfflat_lists=1, ivfflat_probes=1)

    assert params["ivfflat_lists"] == 1
    assert params["ivfflat_probes"] == 1


def test_judge_prompt_version_is_recorded():
    """Two runs graded by different rubrics are not comparable."""
    params = build_params(top_k=4)

    assert params["judge_prompt_version"] == JUDGE_PROMPT_VERSION
    assert params["judge_prompt_version"]


def test_generator_and_judge_are_recorded_separately():
    """A single 'model' field would hide that the judge is independent."""
    params = build_params(top_k=4)

    assert params["generation_model"] != params["judge_model"]


# --------------------------------------------------------------------------
# build_metrics
# --------------------------------------------------------------------------


def test_metrics_cover_accuracy_and_cost():
    metrics = build_metrics(make_result())

    assert set(metrics) == {
        "retrieval_accuracy",
        "retrieval_mrr",
        "rank_1_rate",
        "answer_accuracy",
        "mean_latency_s",
        "mean_tokens_per_query",
        "unparsed_verdicts",
    }


def test_metrics_use_the_derived_accuracies():
    metrics = build_metrics(
        make_result(retrieval_hits=8, answer_hits=10, answerable_total=10)
    )

    assert metrics["retrieval_accuracy"] == 0.8
    assert metrics["answer_accuracy"] == 1.0


def test_hit_rate_and_mrr_are_both_logged():
    """
    The hit rate is the published number and what every earlier run recorded, so
    dropping it would break comparison with them. MRR is the one with
    resolution. Both are needed; neither replaces the other.
    """
    metrics = build_metrics(make_result(ranks=[1, 1, 2, 2]))

    assert metrics["retrieval_accuracy"] == 1.0     # saturated
    assert metrics["retrieval_mrr"] == 0.75         # not saturated
    assert metrics["rank_1_rate"] == 0.5


def test_refusal_accuracy_is_logged_when_measured():
    metrics = build_metrics(make_result(ranks=[1], refusal_hits=4, refusal_total=5))
    assert metrics["refusal_accuracy"] == 0.8


def test_refusal_accuracy_is_omitted_when_not_measured():
    """
    A 0.0 here would read as "never refuses", which is a claim a v1 run never
    tested. Absent is the only honest value.
    """
    metrics = build_metrics(make_result(ranks=[1]))
    assert "refusal_accuracy" not in metrics


def test_chunk_count_is_included_when_known():
    metrics = build_metrics(make_result(ranks=[1]), chunk_count=414)
    assert metrics["chunk_count"] == 414


def test_chunk_count_is_omitted_rather_than_guessed():
    """
    Logging a placeholder chunk count would be worse than logging none: a wrong
    number is indistinguishable from a measured one once it is in the store.
    """
    metrics = build_metrics(make_result())
    assert "chunk_count" not in metrics


@pytest.mark.parametrize("value", [0, 3])
def test_unparsed_verdicts_are_tracked_as_a_metric(value):
    """
    A run degraded by an unreadable judge should be visible as such in the UI,
    not just as a lower accuracy indistinguishable from a genuine one.
    """
    metrics = build_metrics(make_result(unparsed=value))
    assert metrics["unparsed_verdicts"] == value


def test_metric_values_are_all_numeric():
    """MLflow rejects non-numeric metrics; params are where strings belong."""
    for value in build_metrics(make_result(ranks=[1, 2]), chunk_count=414).values():
        assert isinstance(value, (int, float))
