"""
Shared configuration for the RAG project.
Loads secrets/connection info from .env and defines constants used
across the fetch, embed, retrieve, and evaluate scripts.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Postgres connection ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "rag_project"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# --- Embedding model ---
# all-MiniLM-L6-v2 is small, fast, and produces 384-dimensional vectors.
# Good default for a first RAG build; upgrade later if needed.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking ---
CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 50

# --- Retrieval ---
TOP_K = 4  # number of chunks to retrieve per query

# ivfflat partitions to scan per query. The index is sized from the corpus at
# ingestion time (see schema.ivfflat_lists), which at this corpus size yields a
# single partition - so scanning one is exhaustive and exact.
#
# This is not a tuning knob to leave at its default and forget. A fixed
# lists = 10 over ~414 chunks scanned a tenth of the corpus per query and
# dropped the correct chunk on 2 of 10 benchmark questions, which had been
# recorded as a 90% retrieval score. Raise this alongside lists if the corpus
# grows past a few thousand chunks.
IVFFLAT_PROBES = 1

# --- Generation ---
# Small and fast, so the benchmark measures retrieval quality rather than raw
# model horsepower - a large model answers well from pretraining even when
# retrieval misses, which hides exactly what this harness is meant to expose.
#
# This replaced llama-3.1-8b-instant, which produced the first recorded results
# and was later decommissioned by Groq. See the Results section of the README.
GROQ_MODEL = "openai/gpt-oss-20b"

# --- Grading ---
# The judge is deliberately a different, larger model than the generator. A model
# grading its own output has a self-preference bias, and the harness is only
# worth trusting if the grader is independent of the thing being graded.
JUDGE_MODEL = "openai/gpt-oss-120b"

# --- Experiment tracking ---
# SQLite rather than the default ./mlruns file store: MLflow's model registry
# does not work against the file store, and Phase 2 registers the pipeline as a
# versioned pyfunc model. Overridable so a later phase can point at a server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-wikipedia-eval")
MLFLOW_REGISTERED_MODEL = "rag-wikipedia-eval"

# --- Wikipedia topic list (Machine Learning / AI concepts) ---
WIKIPEDIA_ARTICLES = [
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Backpropagation",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Gradient descent",
    "Overfitting",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Natural language processing",
    "Large language model",
]
