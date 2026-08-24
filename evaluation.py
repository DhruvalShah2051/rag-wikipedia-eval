"""
The evaluation harness: benchmark suite and the run loop.

Extracted from 5_evaluate.py so it can be called programmatically - by the
Phase 2 sweep, and later by Phase 4's Airflow quality gate. That script begins
with a digit and cannot be imported.

Scoring is unchanged from Phase 1:
  1. Retrieval accuracy: did the system retrieve a chunk from the correct
     source article?
  2. Answer quality: an LLM judge grades the generated answer against a short
     reference answer, using the explicit ordered rubric in grading.py.

What is new is that each run also reports what it cost - mean latency and mean
tokens per query - so a configuration that scores well but answers slowly is
visible as such rather than looking identical to a fast one.
"""

import json
from dataclasses import asdict, dataclass, field

from config import TOP_K
from grading import grade_answer_with_llm, grade_refusal
from rag_pipeline import answer_question

# Each test case has:
#   - query: the question to ask
#   - expected_source: which Wikipedia article should be retrieved
#   - reference_answer: a short, correct answer used by the LLM judge
#     to grade the generated answer (doesn't need to match wording,
#     just needs to capture the key fact(s))
TEST_CASES = [
    {
        "query": "What is backpropagation used for?",
        "expected_source": "Backpropagation",
        "reference_answer": "It is used to compute gradients for training neural networks by applying the chain rule.",
    },
    {
        "query": "What does a convolutional neural network specialize in processing?",
        "expected_source": "Convolutional neural network",
        "reference_answer": "Image and grid-like data, using convolutional layers to detect spatial patterns.",
    },
    {
        "query": "What is overfitting in machine learning?",
        "expected_source": "Overfitting",
        "reference_answer": "When a model fits the training data too closely, including its noise, and performs poorly on new/unseen data.",
    },
    {
        "query": "What mechanism do transformers use to weigh the importance of different words?",
        "expected_source": "Attention (machine learning)",
        "reference_answer": "The attention mechanism, which assigns different weights to different input tokens.",
    },
    {
        "query": "What type of learning uses labeled data?",
        "expected_source": "Supervised learning",
        "reference_answer": "Supervised learning, which trains on data with labeled input-output pairs.",
    },
    {
        "query": "What algorithm is commonly used to minimize a neural network's loss function?",
        "expected_source": "Gradient descent",
        "reference_answer": "Gradient descent, which iteratively adjusts parameters in the direction that reduces the loss.",
    },
    {
        "query": "What type of neural network is designed for sequential data?",
        "expected_source": "Recurrent neural network",
        "reference_answer": "Recurrent neural networks (RNNs), designed to handle sequential or time-series data.",
    },
    {
        "query": "What is reinforcement learning based on?",
        "expected_source": "Reinforcement learning",
        "reference_answer": "An agent learning to take actions that maximize cumulative reward through trial-and-error interaction with an environment.",
    },
    {
        "query": "What does NLP stand for and what does it deal with?",
        "expected_source": "Natural language processing",
        "reference_answer": "Natural Language Processing; it deals with enabling computers to understand and process human language.",
    },
    {
        "query": "What learning approach works with unlabeled data?",
        "expected_source": "Unsupervised learning",
        "reference_answer": "Unsupervised learning, which finds patterns or structure in data without labeled outputs.",
    },
]

# Questions with no answer anywhere in a corpus of ML/AI articles. The generation
# prompt instructs the model to decline when the context does not contain the
# answer; until these existed, nothing tested whether it actually does.
#
# This is the classic RAG failure mode: the model answers confidently from
# pretraining, the answer is fluent and correct-sounding, and the system looks
# fine while having ignored its own grounding instruction entirely.
#
# Each is deliberately answerable from ordinary pretraining, so a model that
# ignores the context will produce a confident wrong-behaviour answer rather
# than getting stuck - that is exactly what needs to be caught.
REFUSAL_CASES = [
    {"query": "What is the capital city of Australia?", "unanswerable": True},
    {"query": "Who wrote the novel Pride and Prejudice?", "unanswerable": True},
    {"query": "What temperature does water boil at in degrees Celsius?", "unanswerable": True},
    {"query": "How long does it take to bake a loaf of sourdough bread?", "unanswerable": True},
    {"query": "Which team won the 2018 FIFA World Cup final?", "unanswerable": True},
]

# Benchmark suites are versioned because runs scored on different suites are not
# comparable. v1 is frozen: the sweep's six configurations are measured on it,
# and changing it would invalidate the comparison between them.
BENCHMARK_V1_NAME = "v1-10q"
BENCHMARK_V2_NAME = "v2-15q-refusals"

BENCHMARK_V1 = TEST_CASES
BENCHMARK_V2 = TEST_CASES + REFUSAL_CASES

BENCHMARKS = {
    BENCHMARK_V1_NAME: BENCHMARK_V1,
    BENCHMARK_V2_NAME: BENCHMARK_V2,
}


@dataclass
class EvaluationResult:
    """
    One complete pass over the benchmark suite.

    Everything MLflow logs as a metric is on this object, so the tracking layer
    reads from one place rather than recomputing anything.
    """

    top_k: int
    retrieval_hits: int
    answer_hits: int
    answerable_total: int
    mean_latency_s: float
    mean_tokens_per_query: float
    unparsed_verdicts: int
    refusal_hits: int = 0
    refusal_total: int = 0
    benchmark_version: str = BENCHMARK_V1_NAME
    results: list = field(default_factory=list)

    @property
    def total(self):
        return self.answerable_total + self.refusal_total

    @property
    def retrieval_accuracy(self):
        """
        Hit rate: did the expected article appear anywhere in top-k.

        Kept because it is the published number, but it carries one bit of
        information - rank 1 and rank 8 score identically - which is why it
        saturates at 100% on this benchmark. mean_reciprocal_rank is the metric
        with resolution.

        Unanswerable questions are excluded from the denominator: they have no
        correct source, so counting them as misses would understate retrieval
        and counting them as hits would inflate it.
        """
        return self.retrieval_hits / self.answerable_total if self.answerable_total else 0.0

    @property
    def answer_accuracy(self):
        return self.answer_hits / self.answerable_total if self.answerable_total else 0.0

    @property
    def refusal_accuracy(self):
        """
        Fraction of unanswerable questions the model correctly declined.

        None rather than 0.0 when the suite contains no refusal questions - a
        zero would read as "it never refuses", which is a measurement this run
        did not make.
        """
        if not self.refusal_total:
            return None
        return self.refusal_hits / self.refusal_total

    @property
    def _answerable_ranks(self):
        """Rank of the expected article per answerable question; None on a miss."""
        return [r["retrieval_rank"] for r in self.results if not r.get("unanswerable")]

    @property
    def mean_reciprocal_rank(self):
        """
        Mean of 1/rank, counting a miss as 0.

        Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25. Unlike the hit
        rate this distinguishes a pipeline that ranks the right article first
        from one that merely gets it into the window.
        """
        ranks = self._answerable_ranks
        if not ranks:
            return 0.0
        return sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks)

    @property
    def rank_1_rate(self):
        """Fraction of answerable questions whose expected article ranked first."""
        ranks = self._answerable_ranks
        if not ranks:
            return 0.0
        return sum(1 for rank in ranks if rank == 1) / len(ranks)

    def to_dict(self):
        payload = asdict(self)
        payload["total"] = self.total
        payload["retrieval_accuracy"] = self.retrieval_accuracy
        payload["answer_accuracy"] = self.answer_accuracy
        payload["refusal_accuracy"] = self.refusal_accuracy
        payload["mean_reciprocal_rank"] = self.mean_reciprocal_rank
        payload["rank_1_rate"] = self.rank_1_rate
        payload["grading_method"] = "llm_judge"
        return payload

    def save(self, path="evaluation_results.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def run_evaluation(top_k=TOP_K, test_cases=None, benchmark_version=None, verbose=True):
    """
    Run every benchmark question through the pipeline and score it.

    `test_cases` is injectable so tests can drive the loop with a two-question
    suite instead of the real ten. It defaults to benchmark v1, which is frozen:
    the sweep's configurations are all measured on it, so changing the default
    would silently make new runs incomparable with the logged ones.
    """
    test_cases = BENCHMARK_V1 if test_cases is None else test_cases
    benchmark_version = benchmark_version or BENCHMARK_V1_NAME

    results = []
    retrieval_hits = 0
    answer_hits = 0
    answerable_total = 0
    refusal_hits = 0
    refusal_total = 0
    unparsed_verdicts = 0
    latencies = []
    token_counts = []

    for case in test_cases:
        unanswerable = case.get("unanswerable", False)
        result = answer_question(case["query"], top_k=top_k, verbose=False)
        generation = result["generation"]

        retrieved_sources = [c["source_title"] for c in result["retrieved_chunks"]]
        latencies.append(generation.latency_s)
        token_counts.append(generation.total_tokens)

        row = {
            "query": case["query"],
            "unanswerable": unanswerable,
            "retrieved_sources": retrieved_sources,
            "answer": result["answer"],
            "latency_s": round(generation.latency_s, 3),
            "total_tokens": generation.total_tokens,
        }

        if unanswerable:
            # Nothing to retrieve and nothing to compare against: the only
            # correct behaviour is declining. Scored separately from answer
            # accuracy - averaging "declined correctly" together with "answered
            # correctly" would produce a number meaning neither.
            refusal_total += 1
            grading = grade_refusal(case["query"], result["answer"])
            refused = grading["correct"]
            if refused:
                refusal_hits += 1
            if not grading["parsed"]:
                unparsed_verdicts += 1

            row.update({
                "refused": refused,
                "grading_reasoning": grading["reasoning"],
                "retrieval_rank": None,
            })

            if verbose:
                status = "PASS" if refused else "FAIL"
                print(f"[Refusal: {status}] {case['query']}")
                if not refused:
                    print(f"    -> Answered instead of declining: {result['answer'][:120]}")
        else:
            answerable_total += 1

            # Rank of the expected article, not merely whether it appeared. The
            # hit rate cannot tell rank 1 from rank 8, which is why it saturates.
            rank = (
                retrieved_sources.index(case["expected_source"]) + 1
                if case["expected_source"] in retrieved_sources
                else None
            )
            retrieval_correct = rank is not None
            if retrieval_correct:
                retrieval_hits += 1

            grading = grade_answer_with_llm(
                case["query"], case["reference_answer"], result["answer"]
            )
            answer_correct = grading["correct"]
            if answer_correct:
                answer_hits += 1
            if not grading["parsed"]:
                unparsed_verdicts += 1

            row.update({
                "expected_source": case["expected_source"],
                "retrieval_correct": retrieval_correct,
                "retrieval_rank": rank,
                "reference_answer": case["reference_answer"],
                "answer_correct": answer_correct,
                "grading_reasoning": grading["reasoning"],
            })

            if verbose:
                status_r = f"PASS@{rank}" if retrieval_correct else "FAIL"
                status_a = "PASS" if answer_correct else "FAIL"
                print(f"[Retrieval: {status_r} | Answer: {status_a}] {case['query']}")
                if not answer_correct:
                    print(f"    -> Judge reasoning: {grading['reasoning']}")

        results.append(row)

        if verbose and not grading["parsed"]:
            # Scored as a failure, but only because the judge's reply was
            # unreadable. Surface it rather than letting it sink into the
            # accuracy number.
            print(f"    -> WARNING: no verdict found in judge reply: {grading['raw_verdict']!r}")

    total = len(test_cases)
    return EvaluationResult(
        top_k=top_k,
        retrieval_hits=retrieval_hits,
        answer_hits=answer_hits,
        answerable_total=answerable_total,
        refusal_hits=refusal_hits,
        refusal_total=refusal_total,
        benchmark_version=benchmark_version,
        mean_latency_s=sum(latencies) / total if total else 0.0,
        mean_tokens_per_query=sum(token_counts) / total if total else 0.0,
        unparsed_verdicts=unparsed_verdicts,
        results=results,
    )


def print_summary(result):
    """The terminal report."""
    print("\n" + "=" * 64)
    print(f"Benchmark: {result.benchmark_version}  (top_k={result.top_k})")
    print("-" * 64)
    print(f"Retrieval hit rate: {result.retrieval_hits}/{result.answerable_total} "
          f"({result.retrieval_accuracy:.1%})")
    # The metric with resolution. The hit rate above cannot tell rank 1 from
    # rank 8, so it saturates; MRR is what actually distinguishes configurations.
    print(f"Retrieval MRR: {result.mean_reciprocal_rank:.3f} "
          f"(ranked first on {result.rank_1_rate:.0%} of questions)")
    print(f"Answer accuracy (LLM-graded): {result.answer_hits}/{result.answerable_total} "
          f"({result.answer_accuracy:.1%})")
    if result.refusal_total:
        print(f"Refusal accuracy: {result.refusal_hits}/{result.refusal_total} "
              f"({result.refusal_accuracy:.1%})")
    print(f"Mean latency: {result.mean_latency_s:.2f}s per query")
    print(f"Mean tokens: {result.mean_tokens_per_query:.0f} per query")
    if result.unparsed_verdicts:
        print(f"WARNING: {result.unparsed_verdicts} judge reply/replies had no readable "
              f"verdict and were counted as INCORRECT.")
    print("=" * 64)
