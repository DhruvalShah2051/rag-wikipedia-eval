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
from grading import grade_answer_with_llm
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
    total: int
    mean_latency_s: float
    mean_tokens_per_query: float
    unparsed_verdicts: int
    results: list = field(default_factory=list)

    @property
    def retrieval_accuracy(self):
        return self.retrieval_hits / self.total

    @property
    def answer_accuracy(self):
        return self.answer_hits / self.total

    def to_dict(self):
        payload = asdict(self)
        payload["retrieval_accuracy"] = self.retrieval_accuracy
        payload["answer_accuracy"] = self.answer_accuracy
        payload["grading_method"] = "llm_judge"
        return payload

    def save(self, path="evaluation_results.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def run_evaluation(top_k=TOP_K, test_cases=None, verbose=True):
    """
    Run every benchmark question through the pipeline and score it.

    `test_cases` is injectable so tests can drive the loop with a two-question
    suite instead of the real ten.
    """
    test_cases = TEST_CASES if test_cases is None else test_cases

    results = []
    retrieval_hits = 0
    answer_hits = 0
    unparsed_verdicts = 0
    latencies = []
    token_counts = []

    for case in test_cases:
        result = answer_question(case["query"], top_k=top_k, verbose=False)
        generation = result["generation"]

        # Check 1: was the correct source article among the retrieved chunks?
        retrieved_sources = [c["source_title"] for c in result["retrieved_chunks"]]
        retrieval_correct = case["expected_source"] in retrieved_sources

        # Check 2: LLM-graded answer correctness
        grading = grade_answer_with_llm(
            case["query"], case["reference_answer"], result["answer"]
        )
        answer_correct = grading["correct"]

        if retrieval_correct:
            retrieval_hits += 1
        if answer_correct:
            answer_hits += 1
        if not grading["parsed"]:
            unparsed_verdicts += 1

        latencies.append(generation.latency_s)
        token_counts.append(generation.total_tokens)

        results.append({
            "query": case["query"],
            "expected_source": case["expected_source"],
            "retrieved_sources": retrieved_sources,
            "retrieval_correct": retrieval_correct,
            "answer": result["answer"],
            "reference_answer": case["reference_answer"],
            "answer_correct": answer_correct,
            "grading_reasoning": grading["reasoning"],
            "latency_s": round(generation.latency_s, 3),
            "total_tokens": generation.total_tokens,
        })

        if verbose:
            status_r = "PASS" if retrieval_correct else "FAIL"
            status_a = "PASS" if answer_correct else "FAIL"
            print(f"[Retrieval: {status_r} | Answer: {status_a}] {case['query']}")
            if not answer_correct:
                print(f"    -> Judge reasoning: {grading['reasoning']}")
            if not grading["parsed"]:
                # Scored INCORRECT, but only because the judge's reply was
                # unreadable. Surface it rather than letting it sink into the
                # accuracy number.
                print(f"    -> WARNING: no verdict found in judge reply: {grading['raw_verdict']!r}")

    total = len(test_cases)
    return EvaluationResult(
        top_k=top_k,
        retrieval_hits=retrieval_hits,
        answer_hits=answer_hits,
        total=total,
        mean_latency_s=sum(latencies) / total if total else 0.0,
        mean_tokens_per_query=sum(token_counts) / total if total else 0.0,
        unparsed_verdicts=unparsed_verdicts,
        results=results,
    )


def print_summary(result):
    """The terminal report. Unchanged in shape from Phase 1, plus cost."""
    print("\n" + "=" * 60)
    print(f"Retrieval accuracy: {result.retrieval_hits}/{result.total} "
          f"({result.retrieval_accuracy:.1%})")
    print(f"Answer accuracy (LLM-graded): {result.answer_hits}/{result.total} "
          f"({result.answer_accuracy:.1%})")
    print(f"Mean latency: {result.mean_latency_s:.2f}s per query")
    print(f"Mean tokens: {result.mean_tokens_per_query:.0f} per query")
    if result.unparsed_verdicts:
        print(f"WARNING: {result.unparsed_verdicts} judge reply/replies had no readable "
              f"verdict and were counted as INCORRECT.")
    print("=" * 60)
