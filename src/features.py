"""
Feature engineering module for FraudShield-Lite.

Creates 30 features from raw PaySim transaction data, matching the exact
schema used to train the XGBoost model. Features include:

- Balance features (6): diff, ratio, drained, received
- Velocity features (10): step-level + cumulative per origin/destination
- Type encoding (5): one-hot for CASH_IN/OUT, DEBIT, PAYMENT, TRANSFER
- Temporal features (2): hour_of_day, is_night

Input:  subset_500k.csv (or any PaySim-format CSV)
Output: DataFrame with all features + isFraud + isFlaggedFraud
"""

import numpy as np
import pandas as pd
from typing import Optional


def create_all_features(
    df: pd.DataFrame,
    sort: bool = True,
) -> pd.DataFrame:
    """Create all features from raw PaySim data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim data with columns: step, type, amount, nameOrig,
        oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest,
        newbalanceDest, isFraud, isFlaggedFraud
    sort : bool
        Whether to sort by step after feature creation (default True).

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features + isFraud + isFlaggedFraud.
    """
    print("  Creating balance features...")
    df = _create_balance_features(df)

    print("  Creating velocity features...")
    df = _create_velocity_features(df)

    print("  Creating type encoding...")
    df = pd.get_dummies(df, columns=["type"], prefix="tx_type", drop_first=False)

    print("  Creating temporal features...")
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = (((df["step"] % 24) >= 22) | ((df["step"] % 24) <= 5)).astype(int)

    # Drop identifiers
    df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    if sort:
        df = df.sort_values("step").reset_index(drop=True)

    df = df.fillna(0)
    print(f"  Final feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def _create_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Balance-derived features (6 total)."""
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["orig_balance_drained"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)
    df["dest_balance_received"] = (
        df["newbalanceDest"] > df["oldbalanceDest"]
    ).astype(int)
    return df


def _create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity and cumulative features (10 total).

    Uses step-level aggregation and cumulative stats per origin/destination.
    All features are computed from past data only (no future leakage).
    """
    # ── Step-level aggregation (no fraud label used) ──
    step_tx = df.groupby("step").agg(
        step_tx_count=("amount", "count"),
        step_avg_amount=("amount", "mean"),
        step_total_amount=("amount", "sum"),
    ).reset_index()
    df = df.merge(step_tx, on="step", how="left")
    del step_tx

    # ── Per-origin cumulative features ──
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)
    df["orig_tx_cumcount"] = df.groupby("nameOrig").cumcount() + 1
    df["orig_amount_cumsum"] = df.groupby("nameOrig")["amount"].cumsum()
    df["orig_amount_cummean"] = df["orig_amount_cumsum"] / df["orig_tx_cumcount"]

    # ── Per-destination cumulative features ──
    df = df.sort_values(["nameDest", "step"]).reset_index(drop=True)
    df["dest_tx_cumcount"] = df.groupby("nameDest").cumcount() + 1
    df["dest_amount_cumsum"] = df.groupby("nameDest")["amount"].cumsum()

    # ── Amount deviation features ──
    df["amount_dev_from_orig_mean"] = df["amount"] - df["orig_amount_cummean"]
    df["amount_ratio_to_orig_mean"] = df["amount"] / (df["orig_amount_cummean"] + 1)

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names (excludes targets)."""
    return [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balance_diff_orig", "balance_diff_dest",
        "amount_to_orig_ratio", "amount_to_dest_ratio",
        "orig_balance_drained", "dest_balance_received",
        "step_tx_count", "step_avg_amount", "step_total_amount",
        "orig_tx_cumcount", "orig_amount_cumsum", "orig_amount_cummean",
        "dest_tx_cumcount", "dest_amount_cumsum",
        "amount_dev_from_orig_mean", "amount_ratio_to_orig_mean",
        "tx_type_CASH_IN", "tx_type_CASH_OUT", "tx_type_DEBIT",
        "tx_type_PAYMENT", "tx_type_TRANSFER",
        "hour_of_day", "is_night",
    ]
