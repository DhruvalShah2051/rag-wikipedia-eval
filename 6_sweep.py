"""
Step 6: Parameter sweep.

Evaluates the pipeline across a grid of chunk sizes and retrieval depths,
logging each configuration to MLflow and registering it as a model version.

The point is to turn config.py's defaults into measured decisions. TOP_K = 4 and
CHUNK_SIZE_WORDS = 250 were reasonable first guesses that nothing had ever
tested; after this they are either confirmed or replaced by something with a
number behind it.

Ingestion runs once per chunk size, not once per configuration - retrieval depth
does not change the corpus.

Run:
    python 6_sweep.py
"""

import mlflow

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from console import enable_utf8_output
from evaluation import run_evaluation
from experiment import log_evaluation_run
from ingest import ingest_corpus
from rag_model import log_and_register
from schema import ivfflat_lists

CHUNK_SIZES = (250, 500)
TOP_KS = (2, 4, 8)

# Overlap scales with chunk size, keeping the proportion of repeated context
# constant. Sweeping a fixed overlap across different chunk sizes would confound
# the two: at 250 words a 50-word overlap is 20% of a chunk, at 500 it is 10%.
OVERLAP_RATIO = CHUNK_OVERLAP_WORDS / CHUNK_SIZE_WORDS


def sweep():
    rows = []

    try:
        for chunk_size in CHUNK_SIZES:
            overlap = int(chunk_size * OVERLAP_RATIO)

            print(f"\n{'=' * 70}")
            print(f"Ingesting corpus at chunk_size={chunk_size}, overlap={overlap}")
            print("=" * 70)
            chunk_count = ingest_corpus(chunk_size=chunk_size, overlap=overlap, verbose=False)
            lists = ivfflat_lists(chunk_count)
            print(f"{chunk_count} chunks, ivfflat lists={lists}")

            for top_k in TOP_KS:
                print(f"\n--- chunk_size={chunk_size}, top_k={top_k} ---")
                result = run_evaluation(top_k=top_k, verbose=False)

                run_id = log_evaluation_run(
                    result,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    chunk_count=chunk_count,
                    ivfflat_lists=lists,
                    run_name=f"chunk{chunk_size}-topk{top_k}",
                )

                # Register this configuration as a model version from inside the
                # run that measured it, so a registered version always has its
                # evaluation attached.
                with mlflow.start_run(run_id=run_id):
                    log_and_register(top_k=top_k)

                rows.append({
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "chunks": chunk_count,
                    "retrieval": result.retrieval_accuracy,
                    "answer": result.answer_accuracy,
                    "latency": result.mean_latency_s,
                    "tokens": result.mean_tokens_per_query,
                    "run_id": run_id,
                })

                print(f"retrieval {result.retrieval_accuracy:.0%} | "
                      f"answer {result.answer_accuracy:.0%} | "
                      f"{result.mean_latency_s:.2f}s | "
                      f"{result.mean_tokens_per_query:.0f} tokens")

    finally:
        # Always restore the documented baseline, including on Ctrl-C or a
        # mid-sweep failure. Otherwise the database is left holding whichever
        # configuration ran last, and the next person to run 5_evaluate.py
        # measures something they did not ask for.
        print(f"\n{'=' * 70}")
        print(f"Restoring baseline corpus (chunk_size={CHUNK_SIZE_WORDS}, "
              f"overlap={CHUNK_OVERLAP_WORDS})")
        print("=" * 70)
        restored = ingest_corpus(verbose=False)
        print(f"Baseline restored: {restored} chunks")

    return rows


def print_table(rows):
    print(f"\n{'=' * 78}")
    print("SWEEP RESULTS")
    print("=" * 78)
    header = f"{'chunk':>6} {'top_k':>6} {'chunks':>7} {'retrieval':>10} {'answer':>8} {'latency':>9} {'tokens':>8}"
    print(header)
    print("-" * 78)
    for r in rows:
        print(f"{r['chunk_size']:>6} {r['top_k']:>6} {r['chunks']:>7} "
              f"{r['retrieval']:>9.0%} {r['answer']:>8.0%} "
              f"{r['latency']:>8.2f}s {r['tokens']:>8.0f}")
    print("=" * 78)
    print("\nCompare in the UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    enable_utf8_output()
    print_table(sweep())
