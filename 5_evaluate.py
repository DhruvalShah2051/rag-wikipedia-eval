"""
Step 5: Evaluation harness (LLM-graded).

Runs a benchmark suite through the RAG pipeline, scores retrieval and answer
quality, saves the detail to evaluation_results.json, and records the run in
MLflow.

The suite and the run loop live in evaluation.py so they can be called
programmatically - by 6_sweep.py, and later by Phase 4's Airflow quality gate.
This file is the command-line entry point.

    python 5_evaluate.py                                # v1: 10 answerable questions
    python 5_evaluate.py --benchmark v2-15q-refusals    # v2: adds the refusal path

v1 is frozen because every sweep configuration is measured on it. v2 adds five
unanswerable questions, and its refusal_accuracy is reported separately - a run
on one suite is not comparable with a run on the other, which is why
benchmark_version is logged as a parameter.
"""

import argparse

from console import enable_utf8_output
from evaluation import BENCHMARK_V1_NAME, BENCHMARKS, print_summary, run_evaluation
from experiment import log_evaluation_run
from ingest import count_chunks
from schema import ivfflat_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default=BENCHMARK_V1_NAME,
        help="which benchmark suite to score against (default: %(default)s)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="optional MLflow run name, useful for finding the run later",
    )
    args = parser.parse_args()

    enable_utf8_output()

    result = run_evaluation(
        test_cases=BENCHMARKS[args.benchmark],
        benchmark_version=args.benchmark,
    )
    print_summary(result)

    chunk_count = count_chunks()
    run_id = log_evaluation_run(
        result,
        chunk_count=chunk_count,
        ivfflat_lists=ivfflat_lists(chunk_count),
        run_name=args.run_name,
    )

    print("\nDetailed results saved to evaluation_results.json")
    print(f"Logged to MLflow as run {run_id}")
    print("View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
