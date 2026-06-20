"""
Feature engineering module for FraudShield-Lite.

Creates 36 features from raw PaySim transaction data, matching the exact
schema used to train the XGBoost model. Features include:

- Balance features (6): diff, ratio, drained, received
- Velocity features (10): step-level + cumulative per origin/destination
- Rolling velocity features (7): 24h/7d windows + avg time between txs
- Type encoding (5): one-hot for CASH_IN/OUT, DEBIT, PAYMENT, TRANSFER
- Temporal features (2): hour_of_day, is_night
- Raw columns used as features (6): step, amount, oldbalanceOrg,
  newbalanceOrig, oldbalanceDest, newbalanceDest

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

    print("  Creating rolling velocity features (24h/7d)...")
    df = _create_rolling_velocity_features(df)

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


def _create_rolling_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling window velocity features (7 total).

    Uses a merge-based approach (self-join on nameOrig with step-range
    filter) instead of pandas rolling windows, which is both faster and
    memory-efficient for 500K-row datasets.

    New features:
      - tx_count_24h / tx_count_7d: tx count from same nameOrig in last
        24 / 168 steps
      - amt_sum_24h / amt_sum_7d: total amount from same nameOrig in last
        24 / 168 steps
      - amt_mean_24h / amt_mean_7d: avg amount from same nameOrig in last
        24 / 168 steps
      - avg_time_between_tx: average time gap between consecutive
        transactions from the same nameOrig
    """
    df = df.reset_index(drop=True)
    df["_orig_idx"] = df.index

    # Build right-side lookup
    right = df[["nameOrig", "step", "amount"]].copy()
    right = right.rename(columns={"step": "right_step", "amount": "right_amount"})

    # Self-join on nameOrig: find all pairs of transactions with same origin
    joined = df[["_orig_idx", "nameOrig", "step", "amount"]].merge(
        right, on="nameOrig", how="inner"
    )
    # Keep only previous transactions (no future leakage)
    joined = joined[joined["right_step"] < joined["step"]]

    # Window aggregates (24 steps = 24h, 168 steps = 7d)
    windows = [("24h", 24), ("7d", 168)]

    for label, window_size in windows:
        col_count = f"tx_count_{label}"
        col_sum = f"amt_sum_{label}"
        col_mean = f"amt_mean_{label}"

        window_joined = joined[
            joined["right_step"] >= joined["step"] - window_size
        ].copy()

        if len(window_joined) > 0:
            agg = window_joined.groupby("_orig_idx").agg(
                count=("right_amount", "count"),
                sum=("right_amount", "sum"),
            )
            agg[col_count] = agg["count"].astype(np.int32)
            agg[col_sum] = agg["sum"].astype(np.float64)
            agg[col_mean] = np.where(agg["count"] > 0, agg["sum"] / agg["count"], 0.0)

            df = df.merge(
                agg[[col_count, col_sum, col_mean]], on="_orig_idx", how="left"
            )
        else:
            df[col_count] = 0
            df[col_sum] = 0.0
            df[col_mean] = 0.0

        df[col_count] = df[col_count].fillna(0).astype(np.int32)
        df[col_sum] = df[col_sum].fillna(0.0)
        df[col_mean] = df[col_mean].fillna(0.0)

    # avg_time_between_tx: average gap between consecutive txs from same origin
    if len(joined) > 0:
        all_prev = joined.groupby("_orig_idx").agg(
            prev_count=("right_step", "count"),
            last_step=("right_step", "max"),
        )
        df = df.merge(
            all_prev[["prev_count", "last_step"]], on="_orig_idx", how="left"
        )
        df["prev_count"] = df["prev_count"].fillna(0).astype(np.int32)
        df["last_step"] = df["last_step"].fillna(0)

        df["avg_time_between_tx"] = np.where(
            df["prev_count"] >= 2,
            (df["step"] - df["last_step"]) / (df["prev_count"] - 1),
            0.0,
        )
        df = df.drop(columns=["prev_count", "last_step"])
    else:
        df["avg_time_between_tx"] = 0.0

    # Clean up
    df = df.drop(columns=["_orig_idx"])
    del joined, right

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names (excludes targets).

    Returns 36 feature columns: 29 original + 7 rolling velocity features.
    """
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
        # Rolling velocity features (7)
        "tx_count_24h", "tx_count_7d",
        "amt_sum_24h", "amt_sum_7d",
        "amt_mean_24h", "amt_mean_7d",
        "avg_time_between_tx",
    ]
