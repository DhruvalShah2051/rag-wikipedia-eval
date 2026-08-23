"""
Tests for the pyfunc model wrapper.

These cover input handling and configuration capture, not MLflow's own logging
machinery. The input normalisation is worth pinning because a pyfunc model is
called by things that were not written alongside it - the MLflow UI, a serving
process, Phase 3's service - and each hands over a different shape.
"""

import pandas as pd
import pytest

import rag_model
from config import TOP_K
from rag_model import RagPipeline, _as_questions


def test_dataframe_with_question_column():
    frame = pd.DataFrame({"question": ["What is overfitting?", "What is a CNN?"]})
    assert _as_questions(frame) == ["What is overfitting?", "What is a CNN?"]


def test_dataframe_falls_back_to_the_first_column():
    """MLflow's UI sends whatever column the user typed into."""
    frame = pd.DataFrame({"text": ["What is overfitting?"]})
    assert _as_questions(frame) == ["What is overfitting?"]


def test_bare_string_is_wrapped():
    assert _as_questions("What is overfitting?") == ["What is overfitting?"]


def test_list_passes_through():
    assert _as_questions(["a", "b"]) == ["a", "b"]


def test_predict_answers_every_question(monkeypatch):
    asked = []

    def fake_answer_question(query, top_k=TOP_K):
        asked.append((query, top_k))
        return {"answer": f"answer to {query}"}

    monkeypatch.setattr("rag_pipeline.answer_question", fake_answer_question)

    model = RagPipeline(top_k=4)
    answers = model.predict(None, ["q1", "q2"])

    assert answers == ["answer to q1", "answer to q2"]
    assert len(asked) == 2


def test_predict_uses_the_top_k_it_was_registered_with(monkeypatch):
    """
    The captured top_k is the point of versioning a configuration. A loaded
    version must answer at the depth it was registered with, not at whatever
    config.py says at load time.
    """
    used = []

    def fake_answer_question(query, top_k=TOP_K):
        used.append(top_k)
        return {"answer": "..."}

    monkeypatch.setattr("rag_pipeline.answer_question", fake_answer_question)

    RagPipeline(top_k=8).predict(None, ["q"])

    assert used == [8]


def test_default_top_k_comes_from_config():
    assert RagPipeline().top_k == TOP_K


def test_code_paths_cover_the_pipeline_imports():
    """
    A missing module here fails only when a registered version is loaded in a
    fresh process - long after the run that logged it looked successful.
    """
    import inspect

    source = inspect.getsource(rag_model.log_and_register)

    for module in ("rag_pipeline.py", "config.py", "chunking.py", "llm.py"):
        assert module in source
