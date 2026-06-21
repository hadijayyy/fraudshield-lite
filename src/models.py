"""
Model training and evaluation module for FraudShield.

Provides time-based data splitting, cost-sensitive threshold optimisation,
XGBoost training, and comprehensive evaluation routines.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
import xgboost as xgb


# ---------------------------------------------------------------------------
# 1. Time-based train / val / test split
# ---------------------------------------------------------------------------
def time_based_split(
    data: pd.DataFrame,
    time_col: str = "step",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transaction data by TIME STEP BOUNDARY (single hold-out).

    CRITICAL: Must split by unique step values, NOT by row index.
    Otherwise train/val/test can overlap at the same step (e.g. PaySim
    step 12 has 36K rows — a row-index split would let step 12 appear
    in both train and val, causing data leakage).

    Parameters
    ----------
    data : pd.DataFrame
        Sorted or unsorted transaction data. Will be sorted by ``time_col``.
    time_col : str
        Column containing temporal ordering (step order or timestamps).
    test_size : float
        Fraction of the *most recent steps* to hold out for testing.
    val_size : float
        Fraction of *steps* immediately preceding the test set for validation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) in chronological order.
    """
    data = data.sort_values(time_col).reset_index(drop=True)

    # Get unique steps sorted — works with any step range (1-742, 1-744, …)
    unique_steps = sorted(data[time_col].unique())
    n_steps = len(unique_steps)

    # Calculate step boundaries (by count of unique steps, not rows)
    test_start = int(n_steps * (1 - test_size))
    val_start = int(n_steps * (1 - test_size - val_size))

    train_steps = unique_steps[:val_start]
    val_steps = unique_steps[val_start:test_start]
    test_steps = unique_steps[test_start:]

    train_df = data[data[time_col].isin(train_steps)].copy()
    val_df = data[data[time_col].isin(val_steps)].copy()
    test_df = data[data[time_col].isin(test_steps)].copy()

    print(f"  Train: {len(train_df):,} rows, step {train_steps[0]}-{train_steps[-1]} ({len(train_steps)} steps)")
    print(f"  Val:   {len(val_df):,} rows, step {val_steps[0]}-{val_steps[-1]} ({len(val_steps)} steps)")
    print(f"  Test:  {len(test_df):,} rows, step {test_steps[0]}-{test_steps[-1]} ({len(test_steps)} steps)")
    print(f"  No overlap: {set(train_steps).isdisjoint(set(val_steps)) and set(val_steps).isdisjoint(set(test_steps))}")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 1b. Walk-forward cross-validation (multi-fold, step-based)
# ---------------------------------------------------------------------------
def walk_forward_split(
    data: pd.DataFrame,
    time_col: str = "step",
    n_splits: int = 5,
    test_size: float = 0.2,
) -> None:
    """Generate walk-forward train/test splits using step-based TimeSeriesSplit.

    Unlike ``time_based_split`` (which produces a single hold-out set), this
    function yields ``n_splits`` expanding-window folds. Each fold respects
    step boundaries so there is no data leakage between folds.

    The number of available steps is inferred from the data, so this works
    correctly with both the full dataset (744 steps) and the subset (742 steps).

    Parameters
    ----------
    data : pd.DataFrame
        Transaction data containing ``time_col``.
    time_col : str
        Column with temporal step values.
    n_splits : int
        Number of walk-forward folds to generate.
    test_size : float
        Fraction of the *most recent* steps to hold out for testing in the
        final fold. Each earlier fold proportionally smaller.

    Yields
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) for one fold, both chronologically sorted.
    """
    data = data.sort_values(time_col).reset_index(drop=True)
    unique_steps = sorted(data[time_col].unique())
    n_steps = len(unique_steps)

    # How many test steps per fold
    test_steps_count = max(1, int(n_steps * test_size))

    # Use step-index array for TimeSeriesSplit
    step_indices = np.arange(n_steps)
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=None,
        test_size=test_steps_count,
        gap=0,
    )

    folds = list(tscv.split(step_indices))
    print(f"  Walk-forward CV: {n_steps} unique steps, {len(folds)} folds")

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_steps = [unique_steps[i] for i in train_idx]
        test_steps_ = [unique_steps[i] for i in test_idx]

        train_df = data[data[time_col].isin(train_steps)].copy()
        test_df = data[data[time_col].isin(test_steps_)].copy()

        # Check: is test_df non-empty?  TimeSeriesSplit can produce empty
        # test sets when n_steps is small relative to n_splits.
        if len(test_df) == 0:
            continue

        print(
            f"    Fold {fold_idx + 1}: train {len(train_df):,} rows "
            f"(steps {train_steps[0]}-{train_steps[-1]}) → "
            f"test {len(test_df):,} rows "
            f"(steps {test_steps_[0]}-{test_steps_[-1]})"
        )
        yield train_df, test_df


# ---------------------------------------------------------------------------
# 2. Business cost calculation (in Rupiah)
# ---------------------------------------------------------------------------
def calculate_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_cost: float = 500_000.0,
    false_positive_cost: float = 15_000.0,
) -> Dict[str, float]:
    """Compute the total business cost of the model's decisions in Rupiah.

    Fraud costs are incurred when a fraudulent transaction is approved (FN),
    and false-positive costs are incurred when a legitimate transaction is
    unnecessarily reviewed or blocked (FP).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (1 = fraud, 0 = legitimate).
    y_pred : np.ndarray
        Binary predictions (1 = flagged fraud, 0 = approved).
    fraud_cost : float
        Average monetary loss per missed fraudulent transaction (Rupiah).
    false_positive_cost : float
        Average operational cost per false-positive alert (Rupiah).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - fraud_saved_rp: losses prevented by catching fraud (TP * fraud_cost)
        - fraud_missed_rp: losses from fraud that was missed (FN * fraud_cost)
        - fp_cost_rp: operational cost of false positives (FP * false_positive_cost)
        - net_value_rp: fraud_saved_rp - fraud_missed_rp - fp_cost_rp
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fraud_saved_rp = tp * fraud_cost
    fraud_missed_rp = fn * fraud_cost
    fp_cost_rp = fp * false_positive_cost
    net_value_rp = fraud_saved_rp - fraud_missed_rp - fp_cost_rp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "fraud_saved_rp": fraud_saved_rp,
        "fraud_missed_rp": fraud_missed_rp,
        "fp_cost_rp": fp_cost_rp,
        "net_value_rp": net_value_rp,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }


# ---------------------------------------------------------------------------
# 3. Optimal threshold search
# ---------------------------------------------------------------------------
def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fraud_cost: float = 500_000.0,
    false_positive_cost: float = 15_000.0,
) -> Tuple[float, float, Dict[str, Any]]:
    """Find the decision threshold that maximises net business value.

    Evaluates thresholds from 0.1 to 0.9 in steps of 0.01 and returns the
    threshold with the highest net_value_rp.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_prob : np.ndarray
        Predicted fraud probabilities from the model.
    fraud_cost : float
        Cost per missed fraudulent transaction (Rupiah).
    false_positive_cost : float
        Cost per false-positive alert (Rupiah).

    Returns
    -------
    Tuple[float, float, Dict[str, Any]]
        (optimal_threshold, max_net_value, best_metrics).
    """
    best_threshold = 0.5
    best_net_value = -float("inf")
    best_metrics = {}

    thresholds = np.arange(0.10, 0.91, 0.01)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cost_dict = calculate_business_cost(
            y_true, y_pred, fraud_cost=fraud_cost, false_positive_cost=false_positive_cost
        )
        if cost_dict["net_value_rp"] > best_net_value:
            best_net_value = cost_dict["net_value_rp"]
            best_threshold = thresh
            best_metrics = cost_dict

    return best_threshold, best_net_value, best_metrics


# ---------------------------------------------------------------------------
# 4. Train XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping on the validation set.

    Uses scale_pos_weight to handle class imbalance. Prints AUC, F1, and
    recall at each evaluation round.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    x_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation labels.
    params : dict or None
        XGBoost hyperparameters. Falls back to sensible defaults if not provided.

    Returns
    -------
    xgb.XGBClassifier
        Fitted XGBoost model.
    """
    if params is None:
        # Calculate scale_pos_weight from imbalance ratio
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)
        print(f"  scale_pos_weight = {scale_pos_weight:.2f} (neg:{neg_count}, pos:{pos_count})")

        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 1,
            "use_label_encoder": False,
        }

    model = xgb.XGBClassifier(**params)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=True,
        early_stopping_rounds=50,
    )

    return model


# ---------------------------------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------------------------------
def evaluate_model(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run a full evaluation suite on a held-out test set.

    Computes ROC-AUC, precision, recall, F1, confusion matrix, and business
    cost metrics.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost model.
    x_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    threshold : float
        Decision threshold for converting probabilities to binary labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation metrics including auc, f1, precision, recall,
        confusion_matrix, and cost breakdown.
    """
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    cost = calculate_business_cost(y_test, y_pred)

    return {
        "auc": round(auc, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "fraud_saved_rp": cost["fraud_saved_rp"],
        "fraud_missed_rp": cost["fraud_missed_rp"],
        "fp_cost_rp": cost["fp_cost_rp"],
        "net_value_rp": cost["net_value_rp"],
    }


# ---------------------------------------------------------------------------
# 6. Train baseline model (Dummy + Logistic Regression)
# ---------------------------------------------------------------------------
def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train baseline models (DummyClassifier + LogisticRegression) and evaluate.

    Parameters
    ----------
    x_train, y_train : Training data
    x_test, y_test : Test data

    Returns
    -------
    Dict[str, Any]
        Evaluation metrics for both baselines.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    results = {}

    # --- Dummy (strategy='most_frequent') ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(x_train, y_train)
    y_pred_dummy = dummy.predict(x_test)
    y_prob_dummy = dummy.predict_proba(x_test)[:, 1]

    results["dummy"] = {
        "auc": round(roc_auc_score(y_test, y_prob_dummy), 6),
        "f1": round(f1_score(y_test, y_pred_dummy, zero_division=0), 6),
        "precision": round(precision_score(y_test, y_pred_dummy, zero_division=0), 6),
        "recall": round(recall_score(y_test, y_pred_dummy, zero_division=0), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred_dummy).tolist(),
    }

    # --- Logistic Regression (with class_weight='balanced') ---
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    y_prob_lr = lr.predict_proba(x_test)[:, 1]

    results["logistic_regression"] = {
        "auc": round(roc_auc_score(y_test, y_prob_lr), 6),
        "f1": round(f1_score(y_test, y_pred_lr, zero_division=0), 6),
        "precision": round(precision_score(y_test, y_pred_lr, zero_division=0), 6),
        "recall": round(recall_score(y_test, y_pred_lr, zero_division=0), 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    }

    return results
