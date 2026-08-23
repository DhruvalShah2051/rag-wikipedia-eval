"""
The RAG pipeline packaged as an MLflow pyfunc model.

This is what makes the model registry real rather than decorative: each swept
configuration is logged and registered as a version that can be loaded back and
asked a question.

WHAT THIS ARTIFACT IS NOT
-------------------------
It is not self-contained, and claiming otherwise would misrepresent it. The
embeddings live in Postgres, not in the artifact. A loaded version needs:

  - a reachable pgvector instance,
  - a corpus ingested at the chunk_size recorded in that version's parameters,
  - a GROQ_API_KEY in the environment.

What the registry versions is the pipeline *configuration* - retrieval depth,
chunk size, models, index settings - together with the code that runs it. That
is genuinely useful (it is how Phase 3's service will know what to load, and how
a regression gets traced to a configuration) but it is not a self-contained
model file, and the distinction is worth stating plainly.
"""

import mlflow
import pandas as pd

from config import MLFLOW_REGISTERED_MODEL, TOP_K


class RagPipeline(mlflow.pyfunc.PythonModel):
    """
    Answers questions through retrieve-then-generate.

    top_k is captured at log time so a loaded version answers at the retrieval
    depth it was registered with, rather than whatever config.py happens to say
    when it is loaded.
    """

    def __init__(self, top_k=TOP_K):
        self.top_k = top_k

    def predict(self, context, model_input, params=None):
        """
        `model_input` may be a DataFrame with a `question` column, a list of
        strings, or a single string. Returns one answer per question.

        Imported inside the method rather than at module scope: MLflow loads
        this class in a fresh process, and importing the pipeline at module
        level would pull in the embedding model before the loader is ready.
        """
        from rag_pipeline import answer_question

        questions = _as_questions(model_input)
        return [answer_question(q, top_k=self.top_k)["answer"] for q in questions]


def _as_questions(model_input):
    """Normalise the accepted input shapes to a list of question strings."""
    if isinstance(model_input, pd.DataFrame):
        column = "question" if "question" in model_input.columns else model_input.columns[0]
        return model_input[column].tolist()
    if isinstance(model_input, str):
        return [model_input]
    return list(model_input)


def log_and_register(top_k=TOP_K, registered_model_name=MLFLOW_REGISTERED_MODEL):
    """
    Log the pipeline as a pyfunc model on the active run and register a version.

    Must be called inside an active MLflow run, so the registered version is tied
    to the metrics that were measured for that configuration - a registered model
    with no evaluation attached is the thing this phase is meant to avoid.
    """
    return mlflow.pyfunc.log_model(
        name="rag_pipeline",
        python_model=RagPipeline(top_k=top_k),
        input_example=pd.DataFrame({"question": ["What is backpropagation used for?"]}),
        registered_model_name=registered_model_name,
        # The pipeline modules the loaded artifact needs to import.
        code_paths=[
            "rag_pipeline.py",
            "config.py",
            "chunking.py",
            "grading.py",
            "schema.py",
            "llm.py",
        ],
    )
