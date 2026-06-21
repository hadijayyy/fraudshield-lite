"""Tests for feature engineering module."""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.features import create_all_features, get_feature_columns

def _make_raw(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        "step": np.random.randint(1, 744, n),
        "type": np.random.choice(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"], n),
        "amount": np.random.exponential(500000, n),
        "nameOrig": [f"C{i:010d}" for i in range(n)],
        "oldbalanceOrg": np.random.exponential(1000000, n),
        "newbalanceOrig": np.random.exponential(800000, n),
        "nameDest": [f"M{i:010d}" for i in range(n)],
        "oldbalanceDest": np.random.exponential(500000, n),
        "newbalanceDest": np.random.exponential(700000, n),
        "isFraud": np.zeros(n, dtype=int),
        "isFlaggedFraud": np.zeros(n, dtype=int),
    })

def test_create_all_features_returns_dataframe():
    result = create_all_features(_make_raw(500))
    assert isinstance(result, pd.DataFrame) and len(result) == 500

def test_create_all_features_has_correct_columns():
    result = create_all_features(_make_raw(500))
    expected = set(get_feature_columns())
    actual = set(c for c in result.columns if c not in ["isFraud", "isFlaggedFraud"])
    assert expected == actual

def test_no_nulls():
    result = create_all_features(_make_raw(1000))
    assert result.isnull().sum().sum() == 0

def test_single_row():
    result = create_all_features(_make_raw(1))
    assert len(result) == 1 and result.isnull().sum().sum() == 0

def test_extreme_amounts():
    raw = _make_raw(100)
    raw.loc[0, "amount"] = 1e-12
    raw.loc[1, "amount"] = 1e12
    result = create_all_features(raw)
    assert result.isnull().sum().sum() == 0

def test_zero_balances():
    raw = _make_raw(100)
    raw[["oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]] = 0.0
    result = create_all_features(raw)
    assert result.isnull().sum().sum() == 0
