"""
Step 1: Fetch Wikipedia articles on ML/AI topics and save them locally.

Run this first. Output goes into ./data/raw/<article_title>.txt
"""

import os
import wikipediaapi
from config import WIKIPEDIA_ARTICLES

RAW_DATA_DIR = "data/raw"


def fetch_and_save_articles():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Wikipedia's API requires a descriptive user agent identifying your project
    wiki = wikipediaapi.Wikipedia(
        user_agent="RAGLearningProject/1.0 (personal project for portfolio)",
        language="en",
    )

    fetched, skipped = 0, 0

    for title in WIKIPEDIA_ARTICLES:
        page = wiki.page(title)

        if not page.exists():
            print(f"[SKIP] Page not found: {title}")
            skipped += 1
            continue

        # Save the full article text to a .txt file
        safe_filename = title.replace(" ", "_").replace("/", "_") + ".txt"
        filepath = os.path.join(RAW_DATA_DIR, safe_filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.text)

        print(f"[OK] Saved '{title}' ({len(page.text)} chars) -> {filepath}")
        fetched += 1

    print(f"\nDone. Fetched {fetched} articles, skipped {skipped}.")


if __name__ == "__main__":
    fetch_and_save_articles()
