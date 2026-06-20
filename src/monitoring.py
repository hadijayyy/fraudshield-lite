"""
Model monitoring and drift detection module for FraudShield.

Tracks population stability (PSI), model health metrics, and triggers
alerts when significant drift is detected.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between two distributions.

    PSI measures how much the actual score distribution has shifted from the
    expected (reference) distribution. A PSI > 0.2 typically indicates
    significant drift warranting investigation.

    Parameters
    ----------
    expected : np.ndarray
        Reference / training-period scores or predictions.
    actual : np.ndarray
        Current-period scores or predictions.
    bins : int
        Number of equal-width bins to discretise the distributions.

    Returns
    -------
    float
        PSI value. > 0.2 suggests actionable drift.
    """
    # Handle empty arrays
    if expected.size == 0 or actual.size == 0:
        return 0.0

    EPSILON = 1e-6

    # Determine shared bin edges from the overall min/max of both distributions
    combined_min = float(np.min(np.concatenate([expected, actual])))
    combined_max = float(np.max(np.concatenate([expected, actual])))

    # If all values are identical, PSI is 0 (no shift)
    if np.isclose(combined_min, combined_max):
        return 0.0

    edges = np.linspace(combined_min, combined_max, bins + 1)

    # Digitise and count — np.digitise uses rightmost edge as its own bin, so
    # we clip the last bin index to bins-1.
    exp_bins = np.clip(np.digitize(expected, edges, right=False) - 1, 0, bins - 1)
    act_bins = np.clip(np.digitize(actual, edges, right=False) - 1, 0, bins - 1)

    exp_counts = np.bincount(exp_bins, minlength=bins)[:bins]
    act_counts = np.bincount(act_bins, minlength=bins)[:bins]

    exp_pct = exp_counts / expected.size
    act_pct = act_counts / actual.size

    # Add epsilon to avoid log(0) or division by zero
    exp_pct = np.where(exp_pct == 0, EPSILON, exp_pct)
    act_pct = np.where(act_pct == 0, EPSILON, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def check_model_health(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> Dict[str, str]:
    """Compare current model metrics against pre-defined thresholds.

    Each metric is classified as 'healthy', 'warn', or 'critical' based on
    proximity to its threshold.

    Rules
    -----
    - metric >= threshold  → 'healthy'
    - metric >= 0.95 * threshold (within 5% below) → 'warn'
    - metric < 0.95 * threshold → 'critical'

    Parameters
    ----------
    metrics : Dict[str, float]
        Current-period performance metrics (e.g., AUC, precision, recall).
    thresholds : Dict[str, float]
        Corresponding alert thresholds. Keys should match ``metrics``.

    Returns
    -------
    Dict[str, str]
        Status per metric: 'healthy', 'warn', or 'critical'.
    """
    statuses: Dict[str, str] = {}
    for key in metrics:
        threshold_val = thresholds.get(key)
        if threshold_val is None:
            # No threshold defined for this metric → default healthy
            statuses[key] = "healthy"
            continue

        metric_val = metrics[key]
        if metric_val >= threshold_val:
            statuses[key] = "healthy"
        elif metric_val >= 0.95 * threshold_val:
            statuses[key] = "warn"
        else:
            statuses[key] = "critical"
    return statuses


def alert_drift(
    psi: float,
    auc_drop: float,
    psi_threshold: float = 0.2,
    auc_drop_threshold: float = 0.03,
) -> Dict[str, Any]:
    """Generate a drift alert payload if any metric exceeds its threshold.

    Parameters
    ----------
    psi : float
        Current Population Stability Index.
    auc_drop : float
        Drop in AUC relative to the reference period.
    psi_threshold : float
        PSI threshold for triggering an alert (default 0.2).
    auc_drop_threshold : float
        AUC drop threshold for triggering an alert (default 0.03).

    Returns
    -------
    Dict[str, Any]
        Alert payload with ``alert`` (bool), ``reason`` (str), and metric values.
    """
    # TODO: implement alert logic
    psi_triggered = psi > psi_threshold
    auc_triggered = auc_drop > auc_drop_threshold

    if not psi_triggered and not auc_triggered:
        return {
            "alert": False,
            "reason": "No drift detected",
            "severity": None,
            "psi": psi,
            "auc_drop": auc_drop,
            "timestamp": datetime.now().isoformat(),
        }

    # Build reason string and determine severity
    reasons = []
    if psi_triggered:
        reasons.append(f"PSI={psi:.4f} exceeds threshold {psi_threshold} — retrain recommended")
    if auc_triggered:
        reasons.append(f"AUC drop={auc_drop:.4f} exceeds threshold {auc_drop_threshold} — retrain recommended")

    if psi_triggered and auc_triggered:
        severity = "high"
    elif psi_triggered:
        severity = "medium"
    else:
        severity = "low"

    return {
        "alert": True,
        "reason": "; ".join(reasons),
        "severity": severity,
        "psi": psi,
        "auc_drop": auc_drop,
        "timestamp": datetime.now().isoformat(),
    }
