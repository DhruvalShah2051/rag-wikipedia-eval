# Containerized RAG evaluation harness.
#
# The default command runs the benchmark suite against a reachable Postgres, so
# `docker run` reproduces the measurement rather than just starting a shell.
# Phase 3 adds a FastAPI service and will override CMD.

FROM python:3.13-slim

# Unbuffered so evaluation output streams out of the container as it runs
# rather than appearing all at once when the process exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# Non-root user, created before the model is baked in so the cache is written
# somewhere it can actually read at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser "$HF_HOME" /app

COPY requirements.txt .

# torch comes from the CPU index. The default PyPI Linux wheel bundles CUDA and
# would add several GB to an image that never touches a GPU.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

USER appuser

# Bake the embedding weights into the image so a container start does not
# depend on Hugging Face being reachable.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=appuser:appuser *.py ./

# Connection details and the Groq key come from the environment. config.py calls
# load_dotenv(), which is a no-op here since .env is excluded from the build.
CMD ["python", "5_evaluate.py"]
