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
from evaluation import EvaluationResult
from experiment import build_metrics, build_params
from grading import JUDGE_PROMPT_VERSION


def make_result(retrieval_hits=10, answer_hits=9, total=10, unparsed=0):
    return EvaluationResult(
        top_k=4,
        retrieval_hits=retrieval_hits,
        answer_hits=answer_hits,
        total=total,
        mean_latency_s=5.5,
        mean_tokens_per_query=1600.0,
        unparsed_verdicts=unparsed,
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
        "ivfflat_lists",
        "ivfflat_probes",
    }


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
        "answer_accuracy",
        "mean_latency_s",
        "mean_tokens_per_query",
        "unparsed_verdicts",
    }


def test_metrics_use_the_derived_accuracies():
    metrics = build_metrics(make_result(retrieval_hits=8, answer_hits=10, total=10))

    assert metrics["retrieval_accuracy"] == 0.8
    assert metrics["answer_accuracy"] == 1.0


def test_chunk_count_is_included_when_known():
    metrics = build_metrics(make_result(), chunk_count=414)
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
    for value in build_metrics(make_result(), chunk_count=414).values():
        assert isinstance(value, (int, float))
