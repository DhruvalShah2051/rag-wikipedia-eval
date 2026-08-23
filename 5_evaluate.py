"""
Step 5: Evaluation harness (LLM-graded version).

Runs a fixed set of test questions through the RAG pipeline and scores:
  1. Retrieval accuracy: did the system retrieve a chunk from the correct
     source article? (unchanged - simple, reliable check)
  2. Answer quality: an LLM judge (Groq) grades whether the generated
     answer correctly addresses the question, given a short reference
     answer. This replaces brittle keyword matching with semantic
     judgment, so correct-but-differently-worded answers are scored
     fairly.

This is what makes the project a real "harness": reproducible, automated
scoring you can rerun after any pipeline change (different chunk size,
different top_k, different model) to see if accuracy improved or regressed.

This file owns the benchmark suite and the run loop. The rubric itself and the
parsing of the judge's verdict live in grading.py, where they can be imported
and tested.
"""

import json
from rag_pipeline import answer_question
from grading import grade_answer_with_llm

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


def evaluate():
    results = []
    retrieval_hits = 0
    answer_hits = 0
    unparsed_verdicts = 0

    for case in TEST_CASES:
        result = answer_question(case["query"], verbose=False)

        # Check 1: was the correct source article among the retrieved chunks?
        retrieved_sources = [c["source_title"] for c in result["retrieved_chunks"]]
        retrieval_correct = case["expected_source"] in retrieved_sources

        # Check 2: LLM-graded answer correctness (replaces keyword matching)
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

        results.append({
            "query": case["query"],
            "expected_source": case["expected_source"],
            "retrieved_sources": retrieved_sources,
            "retrieval_correct": retrieval_correct,
            "answer": result["answer"],
            "reference_answer": case["reference_answer"],
            "answer_correct": answer_correct,
            "grading_reasoning": grading["reasoning"],
        })

        status_r = "PASS" if retrieval_correct else "FAIL"
        status_a = "PASS" if answer_correct else "FAIL"
        print(f"[Retrieval: {status_r} | Answer: {status_a}] {case['query']}")
        if not answer_correct:
            print(f"    -> Judge reasoning: {grading['reasoning']}")
        if not grading["parsed"]:
            # Scored INCORRECT, but only because the judge's reply was unreadable.
            # Surface it rather than letting it sink into the accuracy number.
            print(f"    -> WARNING: no verdict found in judge reply: {grading['raw_verdict']!r}")

    total = len(TEST_CASES)
    retrieval_accuracy = retrieval_hits / total
    answer_accuracy = answer_hits / total

    print("\n" + "=" * 60)
    print(f"Retrieval accuracy: {retrieval_hits}/{total} ({retrieval_accuracy:.1%})")
    print(f"Answer accuracy (LLM-graded): {answer_hits}/{total} ({answer_accuracy:.1%})")
    if unparsed_verdicts:
        print(f"WARNING: {unparsed_verdicts} judge reply/replies had no readable verdict "
              f"and were counted as INCORRECT.")
    print("=" * 60)

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_accuracy": retrieval_accuracy,
            "answer_accuracy": answer_accuracy,
            "grading_method": "llm_judge",
            "results": results,
        }, f, indent=2)

    print("\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()