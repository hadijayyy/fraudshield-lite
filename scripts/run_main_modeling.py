"""Week 2 — Main Modeling Script (XGBoost + SHAP + Thresholds)"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import json
import os
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/fraudshield')

from src.features import create_balance_features, create_behavioral_features, create_structuring_features
from src.models import time_based_split, calculate_business_cost, find_optimal_threshold, train_xgboost, evaluate_model

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
FIG_DIR = '/home/ubuntu/fraudshield/reports/figures'

# ===== LOAD & FEATURES =====
print("Loading data (200K rows)...")
df = pd.read_csv('/home/ubuntu/fraudshield/data/raw/paysim.csv', nrows=200000)
df = create_balance_features(df)
df = create_behavioral_features(df)
df = create_structuring_features(df)
df['tx_count_total'] = df.groupby('nameOrig')['step'].transform('count')

df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
drop_cols = ['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest', 'type']
feature_cols = [c for c in df_model.columns if c not in drop_cols]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# ===== SPLIT =====
train, val, test = time_based_split(df_model, time_col='step', test_size=0.15, val_size=0.1)

X_train = train[feature_cols].fillna(0).values
y_train = train['isFraud'].values
X_val = val[feature_cols].fillna(0).values
y_val = val['isFraud'].values
X_test = test[feature_cols].fillna(0).values
y_test = test['isFraud'].values

print(f"\nFraud rates: train={y_train.mean():.4f} ({y_train.sum()}), val={y_val.mean():.4f} ({y_val.sum()}), test={y_test.mean():.4f} ({y_test.sum()})")

# ===== IMBALANCE COMPARISON =====
print("\n=== IMBALANCE METHOD COMPARISON ===")
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

# Method 1: scale_pos_weight (default)
print("  Method 1: scale_pos_weight...")
m1 = train_xgboost(X_train, y_train, X_val, y_val)
auc1 = roc_auc_score(y_test, m1.predict_proba(X_test)[:, 1])
print(f"    AUC: {auc1:.4f}")

# Method 2: Undersampling
print("  Method 2: Undersampling...")
fraud_idx = np.where(y_train == 1)[0]
legit_idx = np.where(y_train == 0)[0]
np.random.seed(42)
downsampled = np.random.choice(legit_idx, size=len(fraud_idx), replace=False)
under_idx = np.concatenate([fraud_idx, downsampled])
X_train_under = X_train[under_idx]
y_train_under = y_train[under_idx]
m2_params = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
             "eval_metric": "auc", "early_stopping_rounds": 50, "random_state": 42,
             "verbosity": 0, "use_label_encoder": False}
import xgboost as xgb
m2 = xgb.XGBClassifier(**m2_params)
m2.fit(X_train_under, y_train_under, eval_set=[(X_val, y_val)], verbose=False)
auc2 = roc_auc_score(y_test, m2.predict_proba(X_test)[:, 1])
print(f"    AUC: {auc2:.4f}")

# Method 3: SMOTE (if available)
try:
    from imblearn.over_sampling import SMOTE
    print("  Method 3: SMOTE...")
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    m3 = xgb.XGBClassifier(**m2_params)
    m3.fit(X_sm, y_sm, eval_set=[(X_val, y_val)], verbose=False)
    auc3 = roc_auc_score(y_test, m3.predict_proba(X_test)[:, 1])
    print(f"    AUC: {auc3:.4f}")
    best_method = max([(auc1, 'scale_pos_weight', m1), (auc2, 'undersampling', m2), (auc3, 'SMOTE', m3)])
except ImportError:
    print("  Method 3: SMOTE not available, skipping")
    best_method = max([(auc1, 'scale_pos_weight', m1), (auc2, 'undersampling', m2)])

print(f"\n  Best method: {best_method[1]} (AUC={best_method[0]:.4f})")
model = best_method[2]

# ===== THRESHOLD OPTIMIZATION =====
print("\n=== THRESHOLD OPTIMIZATION ===")
y_prob_test = model.predict_proba(X_test)[:, 1]
best_thresh, best_net, best_metrics = find_optimal_threshold(y_test, y_prob_test)
print(f"  Optimal threshold: {best_thresh:.2f}")
print(f"  Net value: Rp {best_net:,.0f}")

# Threshold plot
thresholds_range = np.arange(0.1, 0.91, 0.01)
costs = []
precisions = []
recalls = []
for t in thresholds_range:
    preds = (y_prob_test >= t).astype(int)
    c = calculate_business_cost(y_test, preds)
    costs.append(c['net_value_rp'])
    precisions.append(c['precision'])
    recalls.append(c['recall'])

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds_range, costs, 'b-', linewidth=2, label='Net Value (Rp)')
ax1.axvline(best_thresh, color='red', linestyle='--', label=f'Optimal ({best_thresh:.2f})')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Net Business Value (Rp)', color='b', fontsize=12)
ax2 = ax1.twinx()
ax2.plot(thresholds_range, precisions, 'g--', alpha=0.7, label='Precision')
ax2.plot(thresholds_range, recalls, 'r--', alpha=0.7, label='Recall')
ax2.set_ylabel('Precision / Recall', fontsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.set_title('Threshold Optimization: Business Value vs Model Metrics', fontsize=14)
plt.savefig(f'{FIG_DIR}/08_threshold_optimization.png')
plt.close()
print("  Saved 08_threshold_optimization.png")

# ===== FINAL TEST EVALUATION =====
print("\n=== FINAL TEST SET EVALUATION ===")
y_pred = (y_prob_test >= best_thresh).astype(int)
final_results = evaluate_model(model, X_test, y_test, threshold=best_thresh)
for k, v in final_results.items():
    if k != 'confusion_matrix':
        print(f"  {k}: {v}")

# Confusion matrix heatmap
cm = np.array(final_results['confusion_matrix'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix (Test Set)', fontsize=14)
plt.savefig(f'{FIG_DIR}/09_confusion_matrix.png')
plt.close()
print("  Saved 09_confusion_matrix.png")

# ROC + PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

from sklearn.metrics import precision_recall_curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUC={final_results["auc"]:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title('ROC Curve')
ax1.legend()

precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob_test)
ax2.plot(recall_arr, precision_arr, 'g-', linewidth=2)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.axhline(y_test.mean(), color='r', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.4f})')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/09_roc_pr_curves.png')
plt.close()
print("  Saved 09_roc_pr_curves.png")

# ===== SHAP EXPLAINABILITY =====
print("\n=== SHAP EXPLAINABILITY ===")
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:1000])

# Summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_cols, show=False, max_display=15)
plt.title('SHAP Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/10_shap_summary.png')
plt.close()
print("  Saved 10_shap_summary.png")

# Feature importance bar
fig, ax = plt.subplots(figsize=(10, 6))
importance = model.feature_importances_
idx = np.argsort(importance)[-15:]
ax.barh(np.array(feature_cols)[idx], importance[idx], color='#2196F3')
ax.set_xlabel('Importance')
ax.set_title('XGBoost Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/11_feature_importance.png')
plt.close()
print("  Saved 11_feature_importance.png")

# ===== BUSINESS IMPACT =====
print("\n=== BUSINESS IMPACT ===")
fraud_caught = final_results.get('fraud_saved_rp', 0)
fp_cost = final_results.get('fp_cost_rp', 0)
net = final_results.get('net_value_rp', 0)
print(f"  Fraud prevented: Rp {fraud_caught:,.0f}")
print(f"  FP review cost: Rp {fp_cost:,.0f}")
print(f"  Net value:      Rp {net:,.0f}")

# Save results
results_summary = {
    'model': 'XGBoost',
    'threshold': best_thresh,
    'auc': final_results['auc'],
    'f1': final_results['f1'],
    'precision': final_results['precision'],
    'recall': final_results['recall'],
    'net_value_rp': net,
    'features': feature_cols,
    'imbalance_method': best_method[1],
}
os.makedirs('/home/ubuntu/fraudshield/reports', exist_ok=True)
with open('/home/ubuntu/fraudshield/reports/model_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print("\n  Saved reports/model_results.json")

print("\n✅ Main modeling complete!")
