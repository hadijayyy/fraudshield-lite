"""
Feedback collection and active learning module for FraudShield.

Logs model predictions, captures confirmed labels from review workflows,
identifies candidates for retraining, and supports periodic model health
checks.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.monitoring import calculate_psi

logger = logging.getLogger(__name__)

# Max size for JSONL log files before rotation (10 MB)
_MAX_LOG_BYTES = 10 * 1024 * 1024


def _rotate_jsonl(log_path: Path) -> None:
    """Rotate a single JSONL log file, keeping up to 5 backups."""
    for i in range(4, 0, -1):
        old = log_path.with_suffix(f".jsonl.{i}")
        if old.exists():
            if i == 4:
                old.unlink()
            else:
                old.rename(log_path.with_suffix(f".jsonl.{i + 1}"))
    log_path.rename(log_path.with_suffix(".jsonl.1"))


def log_prediction(
    prediction_id: str,
    user_id: str,
    transaction_id: str,
    score: float,
    threshold: float,
    decision: str,
    features_hash: Optional[str] = None,
    log_path: str = "logs/predictions.jsonl",
) -> None:
    """Record a model prediction to the feedback log.

    Parameters
    ----------
    prediction_id : str
        Unique identifier for this prediction event.
    user_id : str
        Account or user identifier.
    transaction_id : str
        Transaction identifier.
    score : float
        Model fraud probability score.
    threshold : float
        Decision threshold used.
    decision : str
        Final decision: 'approve', 'review', or 'block'.
    features_hash : str or None
        Optional hash of the feature vector for reproducibility.
    log_path : str
        Path to the JSONL prediction log.
    """
    record = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "transaction_id": transaction_id,
        "score": score,
        "threshold": threshold,
        "decision": decision,
        "features_hash": features_hash,
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Rotate if log exceeds 10 MB
    if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:
        _rotate_jsonl(log_file)

    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Prediction logged: %s -> %s", prediction_id, decision)


def log_confirmed_label(
    prediction_id: str,
    confirmed_label: int,
    reviewer_id: Optional[str] = None,
    log_path: str = "logs/confirmed_labels.jsonl",
) -> None:
    """Record a human-confirmed label after manual review.

    Parameters
    ----------
    prediction_id : str
        Prediction event identifier (must match ``log_prediction``).
    confirmed_label : int
        Human-confirmed label: 1 = fraud, 0 = legitimate.
    reviewer_id : str or None
        Identifier of the human reviewer.
    log_path : str
        Path to the JSONL confirmed-label log.
    """
    record = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confirmed_label": confirmed_label,
        "reviewer_id": reviewer_id,
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Rotate if log exceeds 10 MB
    if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:
        _rotate_jsonl(log_file)

    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Confirmed label logged: %s -> %d", prediction_id, confirmed_label)


def get_retrain_candidates(
    min_samples: int = 1000,
    prediction_log: str = "logs/predictions.jsonl",
    label_log: str = "logs/confirmed_labels.jsonl",
) -> pd.DataFrame:
    """Query prediction + confirmed-label logs for records ready to use
    in a retraining cycle.

    Joins predictions with their confirmed labels and returns a DataFrame
    of records that have been reviewed.

    Parameters
    ----------
    min_samples : int
        Minimum number of confirmed records required before triggering retraining.
    prediction_log : str
        Path to the prediction JSONL log.
    label_log : str
        Path to the confirmed-label JSONL log.

    Returns
    -------
    pd.DataFrame
        Joined prediction/label records with confirmed labels.
    """
    pred_records = []
    try:
        with open(prediction_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    pred_records.append(json.loads(line))
    except FileNotFoundError:
        logger.warning("Prediction log not found: %s", prediction_log)
        return pd.DataFrame()

    label_records = []
    try:
        with open(label_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    label_records.append(json.loads(line))
    except FileNotFoundError:
        logger.warning("Label log not found: %s", label_log)
        return pd.DataFrame()

    if not pred_records or not label_records:
        return pd.DataFrame()

    pred_df = pd.DataFrame(pred_records).drop_duplicates(subset=["prediction_id"])
    label_df = pd.DataFrame(label_records).drop_duplicates(subset=["prediction_id"])

    # Inner join on prediction_id — keeps only records that have a confirmed label
    merged = pd.merge(pred_df, label_df, on="prediction_id", how="inner", validate="one_to_one")

    # Filter to rows where confirmed_label is not null
    df = merged[merged["confirmed_label"].notna()].copy()

    return df


def model_health_check(
    recent_predictions: pd.DataFrame,  # noqa: ARG001 — kept for API compatibility
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> Dict[str, Any]:
    """Run a lightweight health check on the current model using recent
    prediction data and score distributions.

    Compares ``current_scores`` against ``reference_scores`` using the
    Population Stability Index (PSI). Returns a status dict describing
    whether drift has been detected.

    Parameters
    ----------
    recent_predictions : pd.DataFrame
        Recent prediction records (from ``log_prediction``).
    reference_scores : np.ndarray
        Score distribution from the training / reference period.
    current_scores : np.ndarray
        Score distribution from the current period.

    Returns
    -------
    Dict[str, Any]
        Health status, PSI, and any drift flags.
    """
    psi = calculate_psi(reference_scores, current_scores, bins=10)

    # PSI > 0.2 is considered significant drift
    drift_detected = psi > 0.2
    status = "degraded" if drift_detected else "healthy"

    return {
        "status": status,
        "psi": psi,
        "drift_detected": drift_detected,
    }
