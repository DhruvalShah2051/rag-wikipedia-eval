"""
Step 5: Evaluation harness (LLM-graded).

Runs the fixed benchmark suite through the RAG pipeline and scores retrieval
and answer quality, then saves the detail to evaluation_results.json.

The suite and the run loop live in evaluation.py so they can be called
programmatically - by the Phase 2 sweep, and later by Phase 4's Airflow quality
gate. This file is the command-line entry point.
"""

from evaluation import print_summary, run_evaluation

if __name__ == "__main__":
    result = run_evaluation()
    print_summary(result)
    path = result.save()
    print(f"\nDetailed results saved to {path}")
