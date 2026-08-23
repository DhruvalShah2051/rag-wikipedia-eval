"""
Step 4: Interactive query interface.

Run this after chunks are embedded and stored (step 3) to ask
questions against your document set from the command line.
"""

from console import enable_utf8_output
from rag_pipeline import answer_question


def main():
    enable_utf8_output()
    print("RAG Query Interface (ML/AI Wikipedia knowledge base)")
    print("Type a question, or 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = answer_question(query, verbose=True)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
