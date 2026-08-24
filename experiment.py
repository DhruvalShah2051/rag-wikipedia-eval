"""
MLflow instrumentation for the evaluation harness.

Every harness run becomes a tracked experiment, so a result is always
attributable to the configuration that produced it. That is not bookkeeping for
its own sake: this project has now twice published a number that was measuring
something other than what it was named after - once the judge's rubric, once
the vector index's recall - and in both cases the missing information was the
configuration sitting alongside the score.

Payload assembly is deliberately separate from the logging call. build_params()
and build_metrics() are pure dictionary builders that can be unit tested without
a tracking server, which is what keeps the MLflow layer out of CI's way.
"""

import mlflow

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    IVFFLAT_PROBES,
    JUDGE_MODEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from evaluation import BENCHMARK_V1_NAME
from grading import JUDGE_PROMPT_VERSION


def build_params(top_k, chunk_size=CHUNK_SIZE_WORDS, chunk_overlap=CHUNK_OVERLAP_WORDS,
                 ivfflat_lists=None, ivfflat_probes=IVFFLAT_PROBES,
                 benchmark_version=BENCHMARK_V1_NAME):
    """
    Everything that has to match for two runs to be comparable.

    ivfflat_lists and ivfflat_probes are here because of what Phase 2 found: a
    retrieval score is meaningless without the index configuration that produced
    it. Logging the score alone is how the 90% went unquestioned.

    benchmark_version is here for the same reason one level up. Two runs scored
    on different question sets are not comparable at all, and a mixed comparison
    table would look perfectly fine while being meaningless.
    """
    return {
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "generation_model": GROQ_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "benchmark_version": benchmark_version,
        "ivfflat_lists": ivfflat_lists,
        "ivfflat_probes": ivfflat_probes,
    }


def build_metrics(result, chunk_count=None):
    """
    The numbers worth comparing across runs.

    Accuracies come off EvaluationResult as derived properties, so they can never
    disagree with the hit counts they are computed from.

    retrieval_accuracy (the hit rate) and retrieval_mrr are both logged. The hit
    rate is the published number and the one every earlier run recorded, so
    dropping it would break comparison with them - but it carries one bit of
    information and saturates at 100% on this benchmark. MRR is the metric that
    can actually distinguish configurations.
    """
    metrics = {
        "retrieval_accuracy": result.retrieval_accuracy,
        "retrieval_mrr": result.mean_reciprocal_rank,
        "rank_1_rate": result.rank_1_rate,
        "answer_accuracy": result.answer_accuracy,
        "mean_latency_s": result.mean_latency_s,
        "mean_tokens_per_query": result.mean_tokens_per_query,
        "unparsed_verdicts": result.unparsed_verdicts,
    }
    if chunk_count is not None:
        metrics["chunk_count"] = chunk_count
    # Omitted rather than zeroed when the suite has no refusal questions: a 0.0
    # would read as "never refuses", which is a measurement this run did not make.
    if result.refusal_accuracy is not None:
        metrics["refusal_accuracy"] = result.refusal_accuracy
    return metrics


def configure_tracking():
    """Point MLflow at the local SQLite store and the project's experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_evaluation_run(result, chunk_size=CHUNK_SIZE_WORDS, chunk_overlap=CHUNK_OVERLAP_WORDS,
                       chunk_count=None, ivfflat_lists=None,
                       run_name=None, extra_params=None):
    """
    Record one harness run: parameters, metrics, and the detailed results file.

    chunk_size and chunk_overlap are explicit arguments rather than read from
    config, because the sweep changes them per run. Defaulting them to the config
    values would label every swept run with the baseline's chunk size - a run
    tagged with a configuration it did not use is worse than an untagged one.

    Returns the MLflow run id, which the sweep prints so a row in its summary
    table can be traced back to a run in the UI.
    """
    configure_tracking()

    params = build_params(
        top_k=result.top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ivfflat_lists=ivfflat_lists,
        benchmark_version=result.benchmark_version,
    )
    if extra_params:
        params.update(extra_params)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(build_metrics(result, chunk_count=chunk_count))

        # The per-question detail, so a run can be inspected question by question
        # rather than only as an aggregate. This is what makes a regression
        # diagnosable after the fact instead of merely visible.
        path = result.save()
        mlflow.log_artifact(path)

        return run.info.run_id
