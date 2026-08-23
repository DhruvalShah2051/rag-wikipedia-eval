"""
Step 5: Evaluation harness (LLM-graded).

Runs the fixed benchmark suite through the RAG pipeline, scores retrieval and
answer quality, saves the detail to evaluation_results.json, and records the run
in MLflow.

The suite and the run loop live in evaluation.py so they can be called
programmatically - by 6_sweep.py, and later by Phase 4's Airflow quality gate.
This file is the command-line entry point.
"""

from console import enable_utf8_output
from evaluation import print_summary, run_evaluation
from experiment import log_evaluation_run
from ingest import count_chunks
from schema import ivfflat_lists

if __name__ == "__main__":
    enable_utf8_output()

    result = run_evaluation()
    print_summary(result)

    chunk_count = count_chunks()
    run_id = log_evaluation_run(
        result,
        chunk_count=chunk_count,
        ivfflat_lists=ivfflat_lists(chunk_count),
    )

    print(f"\nDetailed results saved to evaluation_results.json")
    print(f"Logged to MLflow as run {run_id}")
    print("View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
