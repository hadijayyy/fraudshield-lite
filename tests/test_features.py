"""
Tests for the feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    create_velocity_features,
    create_balance_features,
    create_behavioral_features,
    create_structuring_features,
)


def test_create_velocity_features_returns_dataframe() -> None:
    """Smoke test: ensure velocity features returns a DataFrame unchanged."""
    df = pd.DataFrame({
        "user_id": ["a", "b"],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "amount": [100.0, 200.0],
    })
    result = create_velocity_features(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_create_balance_features_returns_dataframe() -> None:
    df = pd.DataFrame({
        "user_id": ["a"],
        "balance_before": [5000.0],
        "amount": [100.0],
    })
    result = create_balance_features(df)
    assert isinstance(result, pd.DataFrame)


def test_create_behavioral_features_returns_dataframe() -> None:
    df = pd.DataFrame({
        "user_id": ["a"],
        "merchant_id": ["m1"],
        "amount": [100.0],
    })
    result = create_behavioral_features(df)
    assert isinstance(result, pd.DataFrame)


def test_create_structuring_features_returns_dataframe() -> None:
    df = pd.DataFrame({
        "user_id": ["a", "a"],
        "amount": [9500.0, 9800.0],
    })
    result = create_structuring_features(df)
    assert isinstance(result, pd.DataFrame)
