"""
FraudShield — Full End-to-End Pipeline Script (standalone).

Runs the entire fraud detection pipeline on subset_500k.csv:
  1) Load data
  2) Feature engineering (src/features.py)
  3) Baselines (DummyClassifier + LogisticRegression)
  4) XGBoost training with threshold optimisation
  5) Evaluation with cost matrix
  6) Save metrics to reports/metrics.json
  7) Save figures to figures/

Usage:
    cd /home/ubuntu/fraudshield-lite && python run_pipeline.py
"""

import sys
import os
import json
import time
import warnings

warnings.filterwarnings("ignore")

# Make `src/` importable from this repo
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# ── Project-local imports ──────────────────────────────────────────────
from src.features import create_all_features
from src.models import (
    calculate_business_cost,
    find_optimal_threshold,
    time_based_split,
    train_baseline,
    train_xgboost,
    evaluate_model,
)
from src.utils import load_config

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "subset_500k.csv")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})

# ── Config ─────────────────────────────────────────────────────────────
config = load_config(CONFIG_PATH)
cost_matrix = config.get("cost_matrix", {})
FRAUD_COST = cost_matrix.get("fraud_cost", 500_000)
FP_COST = cost_matrix.get("false_positive_cost", 15_000)
XGB_PARAMS = config.get("model", {}).get("xgboost", {})
print(f"Cost matrix: fraud_cost={FRAUD_COST}, fp_cost={FP_COST}")


def main():
    t_start = time.time()

    # ===================================================================
    # 1. LOAD DATA
    # ===================================================================
    print("=" * 70)
    print("STEP 1: Loading subset_500k.csv ...")
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")
    print(f"   Steps: {df['step'].min()} -> {df['step'].max()} ({df['step'].nunique()} unique)")
    print(f"   Fraud rate: {df['isFraud'].mean():.6f} ({df['isFraud'].sum():,} cases)")

    # ===================================================================
    # 2. FEATURE ENGINEERING
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Feature engineering (velocity, balance, behavioral, structuring)...")
    t1 = time.time()
    df_feat = create_all_features(df)
    print(f"   Done in {time.time() - t1:.1f}s. Shape: {df_feat.shape}")

    # ===================================================================
    # 3. STEP-BASED SPLIT (time_based_split from models.py)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Step-based split (time_based_split)...")
    train_df, val_df, test_df = time_based_split(
        df_feat, time_col="step", test_size=0.2, val_size=0.1
    )

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        fraud_n = split["isFraud"].sum()
        steps = f"step {split['step'].min()}-{split['step'].max()}"
        print(f"   {name}: {len(split):>10,} rows | "
              f"fraud: {fraud_n:>6} ({fraud_n / len(split) * 100:.3f}%) | {steps}")

    # ── Prepare feature matrices ───────────────────────────────────────
    target_col = "isFraud"
    drop_cols = [target_col, "isFlaggedFraud"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    feature_names = feature_cols  # keep for plotting

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    print(f"\n   Feature matrix: {len(feature_cols)} features")
    for name, X_, y_ in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"   X_{name.lower()}: {X_.shape}, "
              f"fraud: {y_.sum()} ({y_.mean()*100:.3f}%)")

    # ===================================================================
    # 4. BASELINES (DummyClassifier + LogisticRegression)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Training baselines...")
    t2 = time.time()
    baseline_results = train_baseline(X_train, y_train, X_test, y_test)

    # Compute business cost from confusion matrices returned by train_baseline
    baseline_cost = {}
    for name in ["dummy", "logistic_regression"]:
        cm = baseline_results[name]["confusion_matrix"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        net_value = tp * FRAUD_COST - fn * FRAUD_COST - fp * FP_COST
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        baseline_cost[name] = {
            "net_value_rp": net_value,
            "fraud_saved_rp": tp * FRAUD_COST,
            "fraud_missed_rp": fn * FRAUD_COST,
            "fp_cost_rp": fp * FP_COST,
            "precision": precision,
            "recall": recall,
        }

    for name, metrics in baseline_results.items():
        net = baseline_cost.get(name, {}).get("net_value_rp", 0)
        print(f"   {name}: AUC={metrics['auc']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"Net=Rp {net:,.0f}")
    print(f"   Baselines done in {time.time() - t2:.1f}s")

    # ===================================================================
    # 5. XGBoost TRAINING
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Training XGBoost...")
    t3 = time.time()

    # Merge train+val for XGB if we want a larger training set, or use
    # train + val separately. We use train only for training, val for
    # early stopping (standard XGB pattern).
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_prob_val)
    print(f"   Validation AUC: {val_auc:.4f}")
    print(f"   Training done in {time.time() - t3:.1f}s")

    # Save trained model
    model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    model.save_model(model_path)
    print(f"   Model saved: {model_path}")

    # ===================================================================
    # 6. THRESHOLD OPTIMISATION (on validation set)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Threshold optimisation on validation set...")
    best_thresh, best_net_val, best_val_metrics = find_optimal_threshold(
        y_val, y_prob_val, fraud_cost=FRAUD_COST, false_positive_cost=FP_COST
    )
    print(f"   Optimal threshold: {best_thresh:.2f}")
    print(f"   Val net value:     Rp {best_net_val:,.0f}")
    print(f"   Val precision:     {best_val_metrics['precision']:.4f}")
    print(f"   Val recall:        {best_val_metrics['recall']:.4f}")

    # ===================================================================
    # 7. FINAL TEST SET EVALUATION
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STEP 7: Final evaluation on TEST set (threshold={best_thresh:.2f})...")
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= best_thresh).astype(int)

    test_auc = roc_auc_score(y_test, y_prob_test)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)
    test_cost = calculate_business_cost(y_test, y_pred_test,
                                         fraud_cost=FRAUD_COST,
                                         false_positive_cost=FP_COST)

    # Also evaluate at threshold=0.5 (default)
    y_pred_default = (y_prob_test >= 0.5).astype(int)
    default_f1 = f1_score(y_test, y_pred_default, zero_division=0)
    default_precision = precision_score(y_test, y_pred_default, zero_division=0)
    default_recall = recall_score(y_test, y_pred_default, zero_division=0)
    default_cm = confusion_matrix(y_test, y_pred_default)
    default_cost = calculate_business_cost(y_test, y_pred_default,
                                            fraud_cost=FRAUD_COST,
                                            false_positive_cost=FP_COST)

    print(f"\n   === TEST SET RESULTS (threshold={best_thresh:.2f}) ===")
    print(f"   AUC:             {test_auc:.6f}")
    print(f"   F1:              {test_f1:.6f}")
    print(f"   Precision:       {test_precision:.6f}")
    print(f"   Recall:          {test_recall:.6f}")
    print(f"   Confusion Matrix: {cm.tolist()}")
    print(f"   Fraud Saved:     Rp {test_cost['fraud_saved_rp']:>12,.0f}")
    print(f"   Fraud Missed:    Rp {test_cost['fraud_missed_rp']:>12,.0f}")
    print(f"   FP Cost:         Rp {test_cost['fp_cost_rp']:>12,.0f}")
    print(f"   Net Value:       Rp {test_cost['net_value_rp']:>12,.0f}")

    # ── Precision@K ────────────────────────────────────────────────────
    print(f"\n   === Precision@K ===")
    sorted_idx = np.argsort(y_prob_test)[::-1]
    p_at_k_results = {}
    for k in [50, 100, 200, 500]:
        top_k_pred = np.zeros_like(y_test)
        top_k_pred[sorted_idx[:k]] = 1
        p_at_k = precision_score(y_test, top_k_pred, zero_division=0)
        fraud_in_k = int(y_test[sorted_idx[:k]].sum())
        p_at_k_results[f"P@{k}"] = {"precision": round(p_at_k, 6), "fraud_caught": fraud_in_k}
        print(f"   P@{k}: {p_at_k:.4f} ({fraud_in_k}/{k} fraud caught)")

    # ── Baseline comparison on same test set ───────────────────────────
    print(f"\n   === BASELINE COMPARISON ON TEST SET ===")
    for name, bm in baseline_results.items():
        cost_b = baseline_cost.get(name, {})
        print(f"   {name}: AUC={bm['auc']:.4f}, F1={bm['f1']:.4f}, "
              f"Recall={bm['recall']:.4f}, Net=Rp {cost_b.get('net_value_rp',0):,.0f}")

    # ===================================================================
    # 8. FIGURES
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 8: Generating figures...")

    # ── 8a. ROC Curve — Baselines + XGBoost ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    # XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_test)
    ax.plot(fpr_xgb, tpr_xgb, color="#E91E63", linewidth=2.5,
            label=f"XGBoost (AUC={test_auc:.4f})")

    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    ax.plot(fpr_lr, tpr_lr, color="#2196F3", linewidth=2,
            label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.4f})")

    # Dummy
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
    fpr_d, tpr_d, _ = roc_curve(y_test, y_prob_dummy)
    ax.plot(fpr_d, tpr_d, color="gray", linewidth=1.5, linestyle="--",
            label=f"Dummy (AUC={roc_auc_score(y_test, y_prob_dummy):.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.2, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(FIG_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"   Saved: {roc_path}")

    # ── 8b. Precision-Recall Curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob_test)
    ax.plot(recall_arr, precision_arr, color="#E91E63", linewidth=2.5,
            label=f"XGBoost (AP~{np.trapz(precision_arr, recall_arr):.4f})")
    ax.axhline(y_test.mean(), color="r", linestyle="--", alpha=0.5,
               label=f"Random baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(FIG_DIR, "pr_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"   Saved: {pr_path}")

    # ── 8c. Confusion Matrix — XGBoost (optimal threshold) ─────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={best_thresh:.2f})", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"   Saved: {cm_path}")

    # ── 8d. Threshold Optimisation Plot ────────────────────────────────
    thresholds_range = np.arange(0.10, 0.91, 0.01)
    net_vals = []
    prec_vals = []
    rec_vals = []
    for t in thresholds_range:
        preds = (y_prob_test >= t).astype(int)
        c = calculate_business_cost(y_test, preds,
                                     fraud_cost=FRAUD_COST,
                                     false_positive_cost=FP_COST)
        net_vals.append(c["net_value_rp"])
        prec_vals.append(c["precision"])
        rec_vals.append(c["recall"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds_range, net_vals, "b-", linewidth=2, label="Net Value (Rp)")
    ax1.axvline(best_thresh, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal ({best_thresh:.2f})")
    ax1.set_xlabel("Decision Threshold", fontsize=12)
    ax1.set_ylabel("Net Business Value (Rp)", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(thresholds_range, prec_vals, "g--", alpha=0.7, linewidth=1.5,
             label="Precision")
    ax2.plot(thresholds_range, rec_vals, "r--", alpha=0.7, linewidth=1.5,
             label="Recall")
    ax2.set_ylabel("Precision / Recall", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)
    ax1.set_title("Threshold Optimisation — Business Value", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    thresh_path = os.path.join(FIG_DIR, "threshold_optimization.png")
    plt.savefig(thresh_path)
    plt.close()
    print(f"   Saved: {thresh_path}")

    # ── 8e. XGBoost Feature Importance (Top 15) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    importance = model.feature_importances_
    top_n = 15
    top_idx = np.argsort(importance)[-top_n:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importance[top_idx]
    ax.barh(range(top_n), top_vals, color="#2196F3", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"XGBoost Feature Importance (Top {top_n})", fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    print(f"   Saved: {fi_path}")

    # ── 8f. Precision@K Chart ──────────────────────────────────────────
    k_range = list(range(10, min(1001, len(y_test)), 10))
    p_at_k_vals = []
    for k in k_range:
        top_k_pred = np.zeros_like(y_test)
        top_k_pred[sorted_idx[:k]] = 1
        p_at_k_vals.append(precision_score(y_test, top_k_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), p_at_k_vals, "b-", linewidth=2, label="XGBoost")
    ax.axhline(y_test.mean(), color="r", linestyle="--", alpha=0.6,
               label=f"Random baseline ({y_test.mean():.4f})")
    ax.set_xlabel("K (top-K transactions flagged)", fontsize=12)
    ax.set_ylabel("Precision@K", fontsize=12)
    ax.set_title("Precision@K — XGBoost Fraud Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pk_path = os.path.join(FIG_DIR, "precision_at_k.png")
    plt.savefig(pk_path)
    plt.close()
    print(f"   Saved: {pk_path}")

    # ── 8g. Baseline Cost Comparison ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_names = list(baseline_cost.keys()) + ["XGBoost"]
    bar_values = [baseline_cost[n]["net_value_rp"] for n in baseline_cost.keys()]
    bar_values.append(test_cost["net_value_rp"])
    bar_colors = []
    for v in bar_values:
        if v < 0:
            bar_colors.append("#F44336")
        else:
            bar_colors.append("#4CAF50")
    bars = ax.bar(bar_names, bar_values, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, bar_values):
        va = "bottom" if val >= 0 else "top"
        offset = 0.02 * max(bar_values) if val >= 0 else -0.02 * max(bar_values)
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"Rp {val:,.0f}", ha="center", va=va, fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net Business Value (Rp)", fontsize=12)
    ax.set_title("Model Comparison — Net Business Value", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    comp_path = os.path.join(FIG_DIR, "model_comparison_cost.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"   Saved: {comp_path}")

    # ===================================================================
    # 9. SAVE ALL METRICS TO reports/metrics.json
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 9: Saving metrics to reports/metrics.json ...")

    # ── Build complete metrics dictionary ───────────────────────────────
    all_metrics = {
        "pipeline_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_duration_s": round(time.time() - t_start, 1),
        "dataset": {
            "source": "subset_500k.csv",
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "num_features": len(feature_cols),
            "feature_names": feature_cols,
            "split": "step-based (train<={}, val={}-{}, test>{})".format(
                train_steps[-1], val_steps[0], val_steps[-1], test_df["step"].min()),
        },
        "cost_matrix": {
            "fraud_cost": FRAUD_COST,
            "false_positive_cost": FP_COST,
        },
        "fraud_rates": {
            "train": round(float(y_train.mean()), 6),
            "val": round(float(y_val.mean()), 6),
            "test": round(float(y_test.mean()), 6),
        },
        "baselines": {
            name: {
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "confusion_matrix": metrics["confusion_matrix"],
                "net_value_rp": baseline_cost.get(name, {}).get("net_value_rp", 0),
            }
            for name, metrics in baseline_results.items()
        },
        "xgboost": {
            "params": XGB_PARAMS,
            "optimal_threshold": round(float(best_thresh), 4),
            "validation": {
                "auc": round(float(val_auc), 6),
                "net_value_rp": round(float(best_net_val), 2),
                "precision": round(float(best_val_metrics["precision"]), 6),
                "recall": round(float(best_val_metrics["recall"]), 6),
            },
            "test": {
                "auc": round(float(test_auc), 6),
                "f1": round(float(test_f1), 6),
                "precision": round(float(test_precision), 6),
                "recall": round(float(test_recall), 6),
                "threshold_optimal": round(float(best_thresh), 4),
                "confusion_matrix_optimal": cm.tolist(),
                "fraud_saved_rp": test_cost["fraud_saved_rp"],
                "fraud_missed_rp": test_cost["fraud_missed_rp"],
                "fp_cost_rp": test_cost["fp_cost_rp"],
                "net_value_rp": test_cost["net_value_rp"],
                "precision_at_k": p_at_k_results,
            },
            "test_default_threshold_0.5": {
                "f1": round(float(default_f1), 6),
                "precision": round(float(default_precision), 6),
                "recall": round(float(default_recall), 6),
                "confusion_matrix": default_cm.tolist(),
                "net_value_rp": default_cost["net_value_rp"],
            },
            "best_iteration": int(model.get_booster().best_iteration)
            if hasattr(model.get_booster(), "best_iteration") and model.get_booster().best_iteration is not None
            else None,
            "feature_importance_top5": [
                {"feature": feature_names[i], "importance": float(importance[i])}
                for i in np.argsort(importance)[-5:][::-1]
            ],
        },
        "figures_generated": [
            "roc_curve.png",
            "pr_curve.png",
            "confusion_matrix.png",
            "threshold_optimization.png",
            "feature_importance.png",
            "precision_at_k.png",
            "model_comparison_cost.png",
        ],
    }

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"   Saved: {metrics_path}")

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE — {elapsed:.0f}s total")
    print(f"{'=' * 70}")
    print(f"Train fraud rate:   {y_train.mean():.6f}")
    print(f"Val fraud rate:     {y_val.mean():.6f}")
    print(f"Test fraud rate:    {y_test.mean():.6f}")
    print(f"XGBoost Test AUC:   {test_auc:.4f}")
    print(f"XGBoost Test F1:    {test_f1:.4f}")
    print(f"Optimal threshold:  {best_thresh:.2f}")
    print(f"Net business value: Rp {test_cost['net_value_rp']:,.0f}")
    print(f"Figures saved to:   {FIG_DIR}/")
    print(f"Metrics saved to:   {metrics_path}")
    print(f"Model saved to:     {model_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
