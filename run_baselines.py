"""
FraudShield-Lite — Baseline Models (Dummy + Logistic Regression)
Compare against XGBoost to show model value.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────
print("=" * 60)
print("FraudShield-Lite — Baseline Models")
print("=" * 60)

df = pd.read_csv("data/features.csv")
train = df[df['step'] <= 600].copy()
test = df[df['step'] > 600].copy()

drop_cols = ['isFraud', 'isFlaggedFraud']
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train[feature_cols], train['isFraud']
X_test, y_test = test[feature_cols], test['isFraud']

print(f"Train: {len(train):,} rows ({y_train.sum():,} fraud, {y_train.mean()*100:.3f}%)")
print(f"Test:  {len(test):,} rows ({y_test.sum():,} fraud, {y_test.mean()*100:.3f}%)")
print(f"Features: {len(feature_cols)}")

# ── Cost Matrix ────────────────────────────────────────────
FRAUD_COST = 500_000   # Rp per missed fraud
FP_COST = 15_000       # Rp per false positive

def calc_net_value(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    net = tp * FRAUD_COST - fn * FRAUD_COST - fp * FP_COST
    return net, tp, fp, fn, tn

results = []

# ── 1. Dummy Classifier (most frequent = all legitimate) ──
print("\n" + "─" * 60)
print("1. Dummy Classifier (predicts all legitimate)")
print("─" * 60)

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print(classification_report(y_test, y_pred_dummy, digits=4, zero_division=0))
net, tp, fp, fn, tn = calc_net_value(y_test, y_pred_dummy)
print(f"Net Value: Rp {net:,.0f}")
print(f"Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
results.append({
    "model": "Dummy (most_frequent)",
    "roc_auc": 0.5,
    "pr_auc": 0.0,
    "f1_fraud": 0.0,
    "precision_fraud": 0.0,
    "recall_fraud": 0.0,
    "net_value_rp": net,
    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
})

# ── 2. Logistic Regression ────────────────────────────────
print("\n" + "─" * 60)
print("2. Logistic Regression")
print("─" * 60)

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

roc_lr = roc_auc_score(y_test, y_proba_lr)
pr_lr = average_precision_score(y_test, y_proba_lr)

print(classification_report(y_test, y_pred_lr, digits=4))
print(f"ROC-AUC: {roc_lr:.4f}")
print(f"PR-AUC:  {pr_lr:.4f}")
net, tp, fp, fn, tn = calc_net_value(y_test, y_pred_lr)
print(f"Net Value: Rp {net:,.0f}")
print(f"Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
results.append({
    "model": "Logistic Regression (balanced)",
    "roc_auc": round(roc_lr, 4),
    "pr_auc": round(pr_lr, 4),
    "f1_fraud": round(classification_report(y_test, y_pred_lr, output_dict=True)['1']['f1-score'], 4),
    "precision_fraud": round(classification_report(y_test, y_pred_lr, output_dict=True)['1']['precision'], 4),
    "recall_fraud": round(classification_report(y_test, y_pred_lr, output_dict=True)['1']['recall'], 4),
    "net_value_rp": net,
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
})

# ── 3. XGBoost (existing model) ───────────────────────────
print("\n" + "─" * 60)
print("3. XGBoost (from trained model)")
print("─" * 60)

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgboost_model.json")

y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)

roc_xgb = roc_auc_score(y_test, y_proba_xgb)
pr_xgb = average_precision_score(y_test, y_proba_xgb)

print(classification_report(y_test, y_pred_xgb, digits=4))
print(f"ROC-AUC: {roc_xgb:.4f}")
print(f"PR-AUC:  {pr_xgb:.4f}")
net, tp, fp, fn, tn = calc_net_value(y_test, y_pred_xgb)
print(f"Net Value: Rp {net:,.0f}")
print(f"Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
results.append({
    "model": "XGBoost (trained)",
    "roc_auc": round(roc_xgb, 4),
    "pr_auc": round(pr_xgb, 4),
    "f1_fraud": round(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['f1-score'], 4),
    "precision_fraud": round(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['precision'], 4),
    "recall_fraud": round(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall'], 4),
    "net_value_rp": net,
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
})

# ── Summary Table ──────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

summary = pd.DataFrame(results)
print(summary[['model', 'roc_auc', 'pr_auc', 'precision_fraud', 'recall_fraud', 'f1_fraud', 'net_value_rp']].to_string(index=False))

# ── Save ───────────────────────────────────────────────────
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'): return obj.item()
        return super().default(obj)

os.makedirs("reports", exist_ok=True)
with open("reports/baseline_comparison.json", "w") as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print("\nSaved to reports/baseline_comparison.json")
