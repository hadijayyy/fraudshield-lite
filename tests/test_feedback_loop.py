"""
Tests for the feedback loop and active learning module.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feedback_collector import (
    log_prediction,
    log_confirmed_label,
    get_retrain_candidates,
    model_health_check,
)


def test_log_prediction_creates_file(tmp_path: Path) -> None:
    """Smoke test: log_prediction writes a JSON line to disk."""
    log_file = tmp_path / "predictions.jsonl"
    log_prediction(
        prediction_id="p001",
        user_id="u1",
        transaction_id="t1",
        score=0.87,
        threshold=0.45,
        decision="review",
        log_path=str(log_file),
    )
    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["prediction_id"] == "p001"


def test_log_confirmed_label_creates_file(tmp_path: Path) -> None:
    log_file = tmp_path / "labels.jsonl"
    log_confirmed_label(
        prediction_id="p001",
        confirmed_label=1,
        reviewer_id="reviewer_01",
        log_path=str(log_file),
    )
    assert log_file.exists()


def test_get_retrain_candidates_returns_dataframe() -> None:
    """Smoke test: get_retrain_candidates returns an empty DataFrame."""
    df = get_retrain_candidates(min_samples=10)
    assert isinstance(df, pd.DataFrame)


def test_model_health_check_returns_dict() -> None:
    result = model_health_check(
        recent_predictions=pd.DataFrame(),
        reference_scores=np.array([0.1, 0.2]),
        current_scores=np.array([0.15, 0.25]),
    )
    assert isinstance(result, dict)
    assert "status" in result
