# RAG Project: ML/AI Wikipedia Knowledge Base

A retrieval-augmented generation (RAG) system that answers questions about
machine learning and AI concepts using a curated set of Wikipedia articles,
Pgvector for semantic search, and Groq (Llama 3) for generation.

This project demonstrates:
- **Retrieval**: chunking, embedding, and vector similarity search
- **Context engineering**: structured prompt assembly with retrieved context
- **Harness engineering**: a reproducible evaluation pipeline with real accuracy metrics

---

## Prerequisites

- Python 3.9+
- PostgreSQL installed locally
- A free Groq API key (https://console.groq.com — sign up, generate a key)

---

## 1. Install PostgreSQL and pgvector

### macOS (Homebrew)
```bash
brew install postgresql@16
brew services start postgresql@16
brew install pgvector
```

### Windows
1. Install Postgres via the installer: https://www.postgresql.org/download/windows/
2. Install pgvector: easiest path is via `vcpkg`, or use the prebuilt binaries
   linked from the pgvector GitHub repo (https://github.com/pgvector/pgvector#windows).
   Alternatively, run Postgres + pgvector inside Docker (see below) to avoid
   Windows build issues entirely.

### Docker (simplest, works the same on any OS)
```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```
This image comes with pgvector pre-installed, no manual extension build needed.

### Create the database
```bash
# Using local Postgres:
createdb rag_project

# Or if using Docker:
docker exec -it rag-postgres createdb -U postgres rag_project
```

---

## 2. Set up the Python environment

```bash
cd rag-project
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `GROQ_API_KEY` — from https://console.groq.com
- `DB_PASSWORD` — whatever password you set for Postgres (matches the
  `POSTGRES_PASSWORD` above if using Docker)
- Adjust `DB_HOST`/`DB_PORT`/`DB_USER` if your setup differs from the defaults

---

## 4. Run the pipeline, in order

```bash
# Step 1: Fetch ~15 Wikipedia articles on ML/AI topics
python 1_fetch_wikipedia.py

# Step 2: Create the Postgres table + pgvector index (run once)
python 2_setup_db.py

# Step 3: Chunk the articles, embed them, and store in Postgres
python 3_chunk_and_embed.py

# Step 4: Ask questions interactively
python 4_query.py

# Step 5: Run the evaluation harness (automated accuracy scoring)
python 5_evaluate.py
```

---

## What each step does

| Step | Script | What it does |
|---|---|---|
| 1 | `1_fetch_wikipedia.py` | Downloads ~15 ML/AI Wikipedia articles as plain text |
| 2 | `2_setup_db.py` | Creates the `document_chunks` table + pgvector index |
| 3 | `3_chunk_and_embed.py` | Splits articles into overlapping chunks, embeds each with `sentence-transformers`, stores vectors in Postgres |
| 4 | `4_query.py` | Interactive CLI: ask a question, see retrieved chunks + generated answer |
| 5 | `5_evaluate.py` | Runs 10 fixed test questions, checks retrieval + answer accuracy, saves results to `evaluation_results.json` |

---

## Customizing / extending this project

- **Change the topic**: edit `WIKIPEDIA_ARTICLES` in `config.py`
- **Tune chunking**: adjust `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` in `config.py`, then re-run steps 2-3
- **Try a different embedding model**: change `EMBEDDING_MODEL_NAME` in `config.py` (make sure `EMBEDDING_DIM` matches the new model's output size)
- **Add more evaluation questions**: extend `TEST_CASES` in `5_evaluate.py`
- **Swap LLM providers**: `rag_pipeline.py` currently uses Groq; swapping to OpenAI/Anthropic just means changing the client initialization and the `generate_answer` call, same pattern as your ProbeLLM multi-provider setup

---

## Resume-ready results

After running `5_evaluate.py`, you'll have real numbers (retrieval accuracy %,
answer accuracy %) and a saved `evaluation_results.json` with full detail on
every test case, exactly what you need to write a legitimate, metric-backed
project bullet instead of a vague "I built a RAG system."
