"""
One-time migration: add rank-aware retrieval metrics to already-logged runs.

Deliberately not a numbered script - it is not a pipeline step, and it should
never need running twice.

The five sweep runs logged before MRR existed still carry the per-question
detail in their `evaluation_results.json` artifact, including which sources came
back in which order. That is everything needed to compute MRR after the fact, so
the earlier runs stay comparable with later ones instead of being stranded on a
metric that was later found to saturate. No model calls, no quota.

Also stamps `benchmark_version` on those runs. They were all measured on the
frozen 10-question v1 suite, but nothing recorded that at the time - and an
unlabelled run is exactly what makes a later comparison table quietly wrong.

    python backfill_mrr.py            # report what would change
    python backfill_mrr.py --apply    # write the metrics back
"""

import argparse
import json

import mlflow

from config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from evaluation import BENCHMARK_V1_NAME


def ranks_from_artifact(payload):
    """
    Recover each question's retrieval rank from a stored results file.

    Older files predate the `retrieval_rank` field, so the rank is recomputed
    from `retrieved_sources` - which was always recorded in retrieval order.
    """
    ranks = []
    for row in payload.get("results", []):
        if row.get("unanswerable"):
            continue
        if row.get("retrieval_rank") is not None:
            ranks.append(row["retrieval_rank"])
            continue
        sources = row.get("retrieved_sources", [])
        expected = row.get("expected_source")
        ranks.append(sources.index(expected) + 1 if expected in sources else None)
    return ranks


def mean_reciprocal_rank(ranks):
    if not ranks:
        return 0.0
    return sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)


def rank_1_rate(ranks):
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r == 1) / len(ranks)


def backfill(apply_changes=False):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        print(f"No experiment named {MLFLOW_EXPERIMENT_NAME!r}; nothing to do.")
        return []

    updated = []

    for run in client.search_runs([experiment.experiment_id], max_results=1000):
        if "retrieval_mrr" in run.data.metrics:
            continue  # already carries the metric

        try:
            path = client.download_artifacts(run.info.run_id, "evaluation_results.json")
        except Exception as exc:
            print(f"  skip {run.info.run_id[:8]}: no results artifact ({type(exc).__name__})")
            continue

        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        ranks = ranks_from_artifact(payload)
        if not ranks:
            print(f"  skip {run.info.run_id[:8]}: no answerable questions in artifact")
            continue

        mrr = mean_reciprocal_rank(ranks)
        first = rank_1_rate(ranks)
        name = run.info.run_name or run.info.run_id[:8]

        distribution = {r: ranks.count(r) for r in sorted(set(ranks), key=lambda x: (x is None, x))}
        print(f"  {name:22s} MRR={mrr:.3f}  rank1={first:.0%}  ranks={distribution}")

        if apply_changes:
            client.log_metric(run.info.run_id, "retrieval_mrr", mrr)
            client.log_metric(run.info.run_id, "rank_1_rate", first)
            if "benchmark_version" not in run.data.params:
                client.log_param(run.info.run_id, "benchmark_version", BENCHMARK_V1_NAME)

        updated.append((name, mrr, first))

    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the metrics back (otherwise report only)")
    args = parser.parse_args()

    print(f"{'Applying' if args.apply else 'Previewing'} backfill "
          f"on experiment {MLFLOW_EXPERIMENT_NAME!r}\n")
    changed = backfill(apply_changes=args.apply)

    print(f"\n{len(changed)} run(s) {'updated' if args.apply else 'would be updated'}.")
    if not args.apply and changed:
        print("Re-run with --apply to write them.")
