# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this project is

A RAG knowledge base with an LLM-graded evaluation harness, currently being extended
with MLOps and infrastructure layers.

This is a portfolio project. Its purpose is to produce **real, demonstrable experience**
with a specific set of tools so they can be honestly claimed on a resume and defended in
a technical interview. That goal shapes every decision below.

GitHub Repository: https://github.com/DhruvalShah2051/rag-wikipedia-eval

## Existing system (already built and working)

- **Ingestion:** 15 Wikipedia AI/ML articles, chunked into 416 chunks
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers, 384 dimensions
- **Vector store:** PostgreSQL with the pgvector extension
- **Generation:** Groq API, `openai/gpt-oss-20b`
- **Evaluation harness:** LLM-as-judge with an explicit rule-based grading rubric over a
  10-question benchmark suite, graded by a separate larger model (`openai/gpt-oss-120b`)
  so the generator is not grading its own output
- **Measured results:** 90% retrieval accuracy at top-4 retrieval depth; answer accuracy
  improved from 50% to 100% across 3 iterative evaluation cycles (keyword matching, then
  a loose LLM judge, then a strict rubric, then a generation-prompt fix)
- **Testing:** 54 pytest tests over chunking, the grading rubric, and retrieval, split into
  unit tests (no database or API key) and integration tests against real pgvector
- **CI:** GitHub Actions runs both tiers on every push and PR and builds the container image;
  GHCR push happens only from `main`

Note on the model: the original results were produced with `llama-3.1-8b-instant`, which
Groq decommissioned during Phase 1 — the API began returning 404 and the harness could not
run at all. The harness was re-run on the models above and reproduced 9/10 and 10/10. Do not
"restore" the Llama model name anywhere; it no longer exists on this account.

Do not change these numbers anywhere in the repo. They came from real runs. If a change
alters measured performance, re-run the evaluation and report the new number rather than
editing the old one.

## Hard rules

1. **Never invent or inflate a claim.** No metric goes into a README, docstring, or commit
   message unless it came from actual terminal output in this repo. If something has not
   been measured, say so rather than estimating.
2. **Never use the word "production"** to describe this system, in code comments, docs, or
   README text. Acceptable verbs: deployed, orchestrated, instrumented, containerized,
   benchmarked. This is a deliberate accuracy constraint, not a style preference.
3. **Never commit secrets.** `.env` is gitignored and stays that way. API keys are read
   from environment variables only.
4. **Do not commit** `venv/`, `data/`, `__pycache__/`, MLflow artifact stores, or anything
   else generated.
5. **Ask before adding a dependency** that is not already in the phase plan below. Scope
   creep is the main risk to this project.

## Environment

Currently running on **Windows (native)**. Phases 1 and 2 are pure Python and YAML and run
fine here.

From **Phase 3 onward the project moves to WSL2**, because Docker, kind/minikube, and
Airflow are all unreliable or unusably slow on native Windows. When that migration happens:

- The repo is cloned fresh into the Linux filesystem (`~/projects/...`), NOT accessed via
  `/mnt/c/...`. Bind-mount and filesystem performance across that boundary is bad.
- The virtualenv is recreated from `requirements.txt`, never copied across.
- Line endings are already normalized via `.gitattributes` (`* text=auto eol=lf`) so shell
  scripts and CI workflows behave the same in both places.

If you are running in WSL and something references a Windows path, flag it rather than
silently rewriting.

## Working style

- **Use Plan mode for each new phase.** Infrastructure work goes wrong at the approach
  level, not the line level. Show the plan before writing files.
- **One phase per branch, one commit series per phase.** Each phase ends with a working,
  committed, pushed state. Session history does not survive the Windows-to-WSL move, so the
  committed repo plus this file is the only handoff.
- **Update the Progress section below** whenever a phase completes.
- Prefer small, readable modules over clever abstraction. This code gets read aloud in
  interviews.
- Every phase that adds a capability also adds at least one test covering it.

## Phase plan

Status key: `TODO` / `IN PROGRESS` / `DONE`

### Phase 1 — Testing and CI/CD (Windows) — DONE
Goal: back the claims "CI/CD pipelines" and "automated testing".
- pytest suite covering retrieval, chunking, and the grading rubric logic
- GitHub Actions workflow: run tests on every push and pull request
- Second workflow job: build the Docker image and push to GitHub Container Registry
- Add a `.gitattributes` with `* text=auto eol=lf` before writing any shell scripts

Things later phases need to know:
- `chunking.py`, `grading.py`, and `schema.py` hold the logic the numbered scripts used to
  contain. The numbered scripts start with a digit and so can never be imported; put
  anything that needs testing or reuse in a module, not in them.
- The embedding model and both Groq clients are lazy. Keep them that way — building them at
  import time is what made the modules unimportable without an API key.
- `5_evaluate.py` still owns the benchmark loop as a script. Phase 2 (MLflow) and Phase 4
  (Airflow quality gate) both need to call it programmatically; extract it to
  `evaluation.py` when the first of those needs it.
- Known, deliberately unfixed: `chunk_text` emits a redundant final chunk when the trailing
  words all fall inside the previous chunk's span. Pinned by a test in
  `tests/test_chunking.py`. Fixing it changes the 416-chunk corpus and every measurement
  over it, so fix it during the Phase 2 chunking sweep, which re-measures anyway.

### Phase 2 — MLflow experiment tracking (Windows) — TODO
Goal: back "MLflow", "experiment tracking", "model registry".
- Instrument the evaluation harness so each run logs to MLflow
- Parameters: chunk size, chunk overlap, top-k, embedding model, judge prompt version
- Metrics: retrieval accuracy, answer accuracy, mean latency, tokens per query
- Run a real sweep (top-k across 2/4/8, at least two chunk sizes) so there is a genuine
  comparison table, not three retroactive data points

### Phase 3 — Kubernetes deployment (WSL2 from here on) — TODO
Goal: back "Kubernetes", "container orchestration".
- Split into two services: FastAPI query service, and Postgres/pgvector
- Manifests: Deployment, Service, ConfigMap, Secret, PersistentVolumeClaim
- Readiness and liveness probes, resource requests and limits
- HorizontalPodAutoscaler, validated against a small load test
- Local `kind` cluster is sufficient. Do not provision paid cloud infrastructure.

### Phase 4 — Airflow orchestration — TODO
Goal: back "Airflow", "workflow orchestration", "pipeline automation".
- DAG: fetch articles, chunk, embed, upsert to pgvector, then trigger the evaluation suite
- Quality gate: the DAG fails if answer accuracy drops below a defined threshold
- Run Airflow via its official Docker Compose setup

### Phase 5 — Observability — TODO
Goal: back "monitoring", "observability", "logging and tracing".
- Prometheus metrics: query latency histogram, retrieval hit rate, token usage, error rate
- Grafana dashboard over those metrics
- Structured JSON logging with a request ID threaded through the retrieval path

### Phase 6 — LangGraph and MCP — TODO
Goal: back "LangGraph", "LangChain", "MCP", "agentic workflows" as genuinely hands-on.
- Rewrite retrieve-and-generate as a LangGraph graph with a routing node that decides
  between answering directly and rewriting the query for a second retrieval pass
- Expose the knowledge base as an MCP server so an MCP client can query it as a tool

### Phase 7 (optional) — Terraform and Kafka — TODO
Only if the first six land cleanly.
- Terraform to provision the local cluster or a small cloud footprint
- Kafka to stream document updates into ingestion instead of batch fetching

## Progress log

Update this as phases complete. Keep entries to one or two lines.

- **Phase 1 — 2026-08-23.** 54 pytest tests (unit + pgvector integration), GitHub Actions CI
  with three jobs, and a container image published to GHCR from `main`.
- **Phase 1, unplanned.** Groq decommissioned `llama-3.1-8b-instant` mid-phase; moved
  generation to `openai/gpt-oss-20b` and split grading onto `openai/gpt-oss-120b`. Re-ran
  the harness: 9/10 retrieval, 10/10 answer accuracy. Also repaired `requirements.txt`,
  whose pins (numpy 1.26.4) could not install on Python 3.13 at all.

## Out of scope

Not part of this project, do not suggest adding them: AWS services, Snowflake, Databricks,
dbt, a frontend in TypeScript or React, or swapping pgvector for Pinecone/Weaviate. Those
are separate gaps addressed elsewhere, and adding them here would dilute the project rather
than strengthen it.
