# RAG Project: ML/AI Wikipedia Knowledge Base

[![CI](https://github.com/DhruvalShah2051/rag-wikipedia-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/DhruvalShah2051/rag-wikipedia-eval/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) system that answers questions about
machine learning and AI concepts using a curated set of Wikipedia articles,
Pgvector for semantic search, and Groq for generation.

This project demonstrates:
- **Retrieval**: chunking, embedding, and vector similarity search
- **Context engineering**: structured prompt assembly with retrieved context
- **Harness engineering**: a reproducible evaluation pipeline with real accuracy metrics
- **Automated testing and CI**: a pytest suite run against a real pgvector instance on every push

---

## Results

Current evaluation across the 10-question test set, over a 414-chunk corpus,
generating with `openai/gpt-oss-20b` and grading with `openai/gpt-oss-120b`:

| Metric | Score |
|---|---|
| Retrieval accuracy | 10/10 (100%) |
| Answer accuracy (LLM-graded) | 10/10 (100%) |
| Mean latency | 5.6s per query |
| Mean tokens | 1629 per query |

Retrieval was previously recorded as 9/10. That number was real but it was not
measuring retrieval — it was measuring the recall of a misconfigured vector
index. See step 7 below; it is the most useful thing this project has found.

**How these numbers were reached (not just reported):**

1. Initial evaluation used keyword-substring matching for answer scoring - fast, but produced false negatives (e.g. penalizing "Supervised learning." for not literally containing the word "label").
2. Switched to an LLM-as-judge grading approach (Groq grades each answer against a short reference answer). Answer accuracy stayed flat at 70%, but manual inspection revealed the judge was **inconsistent**: two structurally identical bare-category answers were graded differently.
3. Rewrote the grading rubric with explicit, ordered rules (e.g. "a bare category name with no supporting mechanism must be marked INCORRECT"). This fixed the inconsistency, but dropped measured accuracy to 50%, revealing that the *generation* prompt, not the evaluation harness, was the real problem: the model was defaulting to terse, unexplained answers.
4. Updated the generation prompt to require 1-2 sentence answers that explain the underlying mechanism, not just name a category. Re-ran evaluation: **answer accuracy rose to 100%**.
5. One retrieval miss appeared to remain (a query about "unlabeled data" didn't retrieve the correct source article), but the model still answered correctly. This was written up as a case study in where RAG helps vs. where it's redundant with pretrained knowledge. **It was not that.** See step 7.
6. Groq later decommissioned `llama-3.1-8b-instant`, and the harness stopped running entirely — a 404 from the API, not a degradation. Generation moved to `openai/gpt-oss-20b` and grading to a separate, larger `openai/gpt-oss-120b`. Splitting the judge off from the generator also removed a weakness that had been there from the start: the model had been grading its own answers.
7. **The retrieval score was measuring the wrong thing.** pgvector's `ivfflat` is an *approximate* index: it partitions vectors into `lists` and scans only `probes` of them per query. The index had been created once with a hardcoded `lists = 10` over ~414 chunks, and `probes` defaults to 1 — so every query examined roughly a tenth of the corpus. Re-ingesting made it worse: `TRUNCATE` and reload left the centroids describing rows that no longer existed. pgvector raises no error in either case; it just returns different neighbours. Measured directly on the same corpus:

   | Search mode | Retrieval accuracy |
   |---|---|
   | `probes=1` (the default, what had been running) | 8/10 |
   | `probes=10` (`probes` == `lists`) | 10/10 |
   | Exact search, index bypassed | 10/10 |

   Retrieval had always been 10/10. Sizing `lists` from the row count (pgvector's own guidance is `rows / 1000`), rebuilding the index after every ingestion, and setting `probes` explicitly per query brought the measured number to what it had actually been all along. The "interesting case study" in step 5 was a misconfigured index.

This progression (build → measure → find a harness bug → fix rubric → find a real generation bug → fix prompt → re-measure) is the actual point of the project: the evaluation harness isn't just a pass/fail gate, it's a debugging tool. Step 6 is the same idea applied to an external change. Step 7 is the sharpest version of it — a headline number was measuring infrastructure configuration rather than the thing it was named after, and only a reproducible harness could show that.

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

# Step 6: Sweep chunk size and retrieval depth, logging each run to MLflow
python 6_sweep.py
```

---

## What each step does

| Step | Script | What it does |
|---|---|---|
| 1 | `1_fetch_wikipedia.py` | Downloads ~15 ML/AI Wikipedia articles as plain text |
| 2 | `2_setup_db.py` | Creates the `document_chunks` table + pgvector index |
| 3 | `3_chunk_and_embed.py` | Splits articles into overlapping chunks, embeds each with `sentence-transformers`, stores vectors in Postgres |
| 4 | `4_query.py` | Interactive CLI: ask a question, see retrieved chunks + generated answer |
| 5 | `5_evaluate.py` | Runs 10 fixed test questions, checks retrieval + answer accuracy, saves results to `evaluation_results.json`, logs the run to MLflow |
| 6 | `6_sweep.py` | Evaluates a grid of chunk sizes and retrieval depths, logging and registering each configuration |

The numbered scripts are entry points. The logic they share lives in modules
they import, which is also what makes it testable — a filename beginning with a
digit is not a valid Python module name, so nothing inside `3_chunk_and_embed.py`
or `5_evaluate.py` can be imported by a test.

| Module | Holds |
|---|---|
| `config.py` | All tunable constants and connection settings |
| `rag_pipeline.py` | Retrieval, prompt assembly, generation |
| `chunking.py` | The chunking rule (`chunk_text`) |
| `grading.py` | The LLM-judge rubric and verdict parsing |
| `schema.py` | The `document_chunks` DDL and ivfflat index sizing |
| `ingest.py` | Chunk, embed, store, and rebuild the index |
| `evaluation.py` | The benchmark suite and the scoring loop |
| `llm.py` | The shared Groq call path — retries and timing |
| `experiment.py` | MLflow parameter and metric logging |
| `rag_model.py` | The pipeline packaged as a pyfunc model |
| `console.py` | UTF-8 console output for the entry points |

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is in two tiers:

- **Unit tests** fake Postgres and the embedding model, so they need no
  database, no API key, and no model download.
- **Integration tests** (marked `integration`) run the real embedding model
  against a real pgvector instance. They skip themselves when no database is
  reachable, so plain `pytest` is safe to run anywhere.

```bash
pytest -m "not integration"   # unit tests only
pytest -m integration         # requires Postgres + pgvector
```

Integration tests build their tables in a dedicated `rag_test` schema, so they
cannot touch the real corpus in `public`.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it does |
|---|---|
| `unit` | Unit tests, and a check that the modules import without a `GROQ_API_KEY` |
| `integration` | Integration tests against a `pgvector/pgvector:pg16` service container |
| `docker` | Builds the image; pushes to GHCR only from `main` |

No job needs a repository secret — every LLM call in the suite is stubbed, and
the registry push uses the built-in `GITHUB_TOKEN`.

---

## Experiment tracking

Every harness run is logged to MLflow, backed by a local SQLite store:

```bash
python 5_evaluate.py                                    # logs one run
mlflow ui --backend-store-uri sqlite:///mlflow.db       # then open localhost:5000
```

Each run records nine parameters and six metrics:

| Parameters | Metrics |
|---|---|
| `top_k`, `chunk_size`, `chunk_overlap` | `retrieval_accuracy`, `answer_accuracy` |
| `embedding_model`, `generation_model`, `judge_model` | `mean_latency_s`, `mean_tokens_per_query` |
| `judge_prompt_version`, `ivfflat_lists`, `ivfflat_probes` | `chunk_count`, `unparsed_verdicts` |

The parameter list is not generic. `judge_prompt_version` is there because two
runs graded under different rubrics are not comparable. `ivfflat_lists` and
`ivfflat_probes` are there because of step 7 above — a retrieval score without
the index configuration behind it is not interpretable, and that exact omission
is what let a wrong number stand. The per-question `evaluation_results.json` is
attached to each run as an artifact, so a regression can be diagnosed question
by question rather than only seen as a lower aggregate.

### Sweep results

`6_sweep.py` evaluates a grid of chunk sizes and retrieval depths. Measured:

| chunk_size | top_k | chunks | retrieval | answer | tokens/query |
|---:|---:|---:|---:|---:|---:|
| 250 | 2 | 414 | 100% | 100% | 935 |
| 250 | 4 | 414 | 100% | 100% | 1622 |
| 250 | 8 | 414 | 100% | 100% | 2955 |
| 500 | 2 | 209 | 100% | 100% | 1607 |
| 500 | 4 | 209 | 100% | 100% | 2930 |
| 500 | 8 | 209 | *not run* | | |

**What this says.** Accuracy is saturated: every configuration scores 100% on
both metrics, so on this benchmark neither chunk size nor retrieval depth
distinguishes them. Cost is not saturated — tokens per query scale roughly
linearly with retrieved context, and `chunk_size=250, top_k=2` answers every
question correctly on **935 tokens**, less than a third of what `top_k=8` spends.

That makes the default `TOP_K = 4` defensible but not optimal for this corpus:
top-2 is measurably cheaper at identical accuracy. The honest caveat is that a
10-question benchmark saturating at 100% cannot distinguish configurations, so
this argues for a harder benchmark before treating it as a tuning result.

**Latency is deliberately omitted from this table.** It is logged as a metric,
but Groq queues requests server-side as an account approaches its quota, and the
sweep ran into that: the first configuration averaged 0.35s per query while
later ones averaged 9–20s at similar token counts. That measures how throttled
the account was, not how the configuration performs, and presenting it as a
property of the configuration would repeat exactly the mistake in step 7 above.

The `500 / 8` cell is unrun: the sweep exhausted the Groq free tier's daily
token cap (200,000 TPD) partway through. Re-running `6_sweep.py` after the quota
resets completes only the missing cell — the sweep skips configurations already
logged in MLflow.

### Model registry

Each swept configuration is logged as an `mlflow.pyfunc` model and registered as
a version of `rag-wikipedia-eval`, from inside the run that measured it — so a
registered version always has its evaluation attached.

**The registered artifact is not self-contained.** The embeddings live in
Postgres, not in the artifact. Loading a version requires a reachable pgvector
instance holding a corpus ingested at that version's `chunk_size`, plus a
`GROQ_API_KEY`. What the registry versions is the pipeline *configuration* and
the code that runs it, which is what Phase 3's service will load and what makes
a regression traceable to a configuration.

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/rag-wikipedia-eval/1")
model.predict(["What is backpropagation used for?"])
```

---

## Running the harness in a container

The image is built by CI and published to GHCR on every push to `main`:

```bash
docker pull ghcr.io/dhruvalshah2051/rag-wikipedia-eval:latest
```

Or build it yourself:

```bash
docker build -t rag-eval .
```

Either way, point it at a reachable Postgres:

```bash
docker run --rm \
  -e GROQ_API_KEY \
  -e DB_HOST=host.docker.internal \
  -e DB_PASSWORD=yourpassword \
  ghcr.io/dhruvalshah2051/rag-wikipedia-eval:latest
```

The default command runs the evaluation harness, so the container reproduces the
benchmark against a reachable Postgres. The embedding weights are baked into the
image, so startup does not depend on Hugging Face being reachable.

---

## Customizing / extending this project

- **Change the topic**: edit `WIKIPEDIA_ARTICLES` in `config.py`
- **Tune chunking**: adjust `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` in `config.py`, then re-run steps 2-3
- **Try a different embedding model**: change `EMBEDDING_MODEL_NAME` in `config.py` (make sure `EMBEDDING_DIM` matches the new model's output size)
- **Add more evaluation questions**: extend `TEST_CASES` in `5_evaluate.py`
- **Change the generator or the judge**: `GROQ_MODEL` and `JUDGE_MODEL` in `config.py`. Keep them different — a model grading its own output grades itself generously
- **After any of the above**: re-run `pytest`, then re-run `5_evaluate.py` and report the number it prints rather than the one already written down
- **Swap LLM providers**: `rag_pipeline.py` currently uses Groq; swapping to OpenAI/Anthropic just means changing the client initialization and the `generate_answer` call, same pattern as your ProbeLLM multi-provider setup
