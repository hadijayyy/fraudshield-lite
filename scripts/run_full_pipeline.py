"""Full Pipeline — 6.3M rows, step-based split, no data leakage"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys, warnings, json, time, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/fraudshield')

from src.features import create_balance_features, create_behavioral_features, create_structuring_features
from src.models import time_based_split, calculate_business_cost, find_optimal_threshold, train_baseline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
FIG = '/home/ubuntu/fraudshield/reports/figures'
os.makedirs(FIG, exist_ok=True)

# ===== 1. LOAD FULL DATASET =====
t0 = time.time()
print("=" * 70)
print("STEP 1: Loading FULL PaySim dataset (6.3M rows)...")
df = pd.read_csv('/home/ubuntu/fraudshield/data/raw/paysim.csv')
print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols in {time.time()-t0:.1f}s")
print(f"   Steps: {df['step'].min()} → {df['step'].max()} ({df['step'].nunique()} unique)")

# ===== 2. FEATURE ENGINEERING =====
t1 = time.time()
print(f"\nSTEP 2: Feature engineering...")
df = create_balance_features(df)
df = create_behavioral_features(df)
df = create_structuring_features(df)
df['tx_count_total'] = df.groupby('nameOrig')['step'].transform('count')
print(f"   Done in {time.time()-t1:.1f}s. Shape: {df.shape}")

# ===== 3. FILTER =====
df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
print(f"\nSTEP 3: Filtered to TRANSFER+CASH_OUT: {df_model.shape[0]:,} rows")

# ===== 4. STEP-BASED SPLIT (NO OVERLAP) =====
print(f"\nSTEP 4: Step-based split (no data leakage)...")
train = df_model[df_model['step'] <= 500].copy()
val   = df_model[(df_model['step'] > 500) & (df_model['step'] <= 620)].copy()
test  = df_model[df_model['step'] > 620].copy()

assert len(set(train['step']) & set(val['step'])) == 0, "LEAKAGE: train-val overlap!"
assert len(set(val['step']) & set(test['step'])) == 0, "LEAKAGE: val-test overlap!"

for name, split in [('Train', train), ('Val', val), ('Test', test)]:
    fraud_n = split['isFraud'].sum()
    steps = f"step {split['step'].min()}-{split['step'].max()}"
    print(f"   {name}: {len(split):>10,} rows | fraud: {fraud_n:>6} ({fraud_n/len(split)*100:.3f}%) | {steps}")

print(f"\n   ✅ Zero step overlap verified!")

# ===== 5. FEATURE MATRIX =====
feature_cols = [c for c in df_model.columns if c not in ['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest', 'type']]
print(f"\nSTEP 5: Feature matrix ({len(feature_cols)} features)")

X_train = train[feature_cols].fillna(0).values
y_train = train['isFraud'].values
X_val   = val[feature_cols].fillna(0).values
y_val   = val['isFraud'].values
X_test  = test[feature_cols].fillna(0).values
y_test  = test['isFraud'].values

# ===== 6. BASELINES =====
print(f"\nSTEP 6: Training baselines...")
baseline_results = train_baseline(X_train, y_train, X_test, y_test)

for name, metrics in baseline_results.items():
    print(f"   {name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

# ===== 7. XGBOOST + SMOTE =====
print(f"\nSTEP 7: Training XGBoost with SMOTE...")
import xgboost as xgb
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"   SMOTE: {len(X_train):,} → {len(X_sm):,} samples")

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=1,  # SMOTE handles imbalance
    eval_metric='auc', early_stopping_rounds=50,
    random_state=42, verbosity=0, use_label_encoder=False
)
model.fit(X_sm, y_sm, eval_set=[(X_val, y_val)], verbose=False)

# Quick check
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"   AUC — Train: {train_auc:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}")

# ===== 8. THRESHOLD OPTIMIZATION =====
print(f"\nSTEP 8: Threshold optimization on validation set...")
y_prob_val = model.predict_proba(X_val)[:, 1]
best_thresh, best_net, best_metrics = find_optimal_threshold(y_val, y_prob_val)
print(f"   Optimal threshold: {best_thresh:.2f}")
print(f"   Val net value: Rp {best_net:,.0f}")

# ===== 9. FINAL TEST EVALUATION =====
print(f"\nSTEP 9: Final evaluation on TEST set...")
y_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= best_thresh).astype(int)

final_metrics = {
    'auc': round(roc_auc_score(y_test, y_prob_test), 4),
    'f1': round(f1_score(y_test, y_pred_test, zero_division=0), 4),
    'precision': round(precision_score(y_test, y_pred_test, zero_division=0), 4),
    'recall': round(recall_score(y_test, y_pred_test, zero_division=0), 4),
}
cost = calculate_business_cost(y_test, y_pred_test)
final_metrics.update(cost)

print("   === TEST SET RESULTS ===")
for k, v in final_metrics.items():
    print(f"   {k}: {v}")

# Precision@K
print(f"\n   === Precision@K ===")
sorted_idx = np.argsort(y_prob_test)[::-1]
for k in [50, 100, 200, 500]:
    top_k_pred = np.zeros_like(y_test)
    top_k_pred[sorted_idx[:k]] = 1
    p_at_k = precision_score(y_test, top_k_pred, zero_division=0)
    fraud_in_k = y_test[sorted_idx[:k]].sum()
    print(f"   P@{k}: {p_at_k:.4f} ({fraud_in_k}/{k} fraud caught)")

# ===== 10. FIGURES =====
print(f"\nSTEP 10: Generating figures...")

# 7: Baseline ROC
fig, ax = plt.subplots(figsize=(8, 6))
for name, color in [('dummy', 'gray'), ('logistic_regression', '#2196F3')]:
    if name == 'dummy':
        m = DummyClassifier(strategy='most_frequent', random_state=42)
    else:
        m = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    m.fit(X_train, y_train)
    y_prob = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.4f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Baseline ROC Curves')
ax.legend(); plt.savefig(f'{FIG}/07_baseline_roc.png'); plt.close()
print("   Saved 07_baseline_roc.png")

# 8: Threshold optimization
thresholds_range = np.arange(0.1, 0.91, 0.01)
costs, precs, recs = [], [], []
for t in thresholds_range:
    preds = (y_prob_test >= t).astype(int)
    c = calculate_business_cost(y_test, preds)
    costs.append(c['net_value_rp'])
    precs.append(c['precision'])
    recs.append(c['recall'])
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds_range, costs, 'b-', lw=2, label='Net Value (Rp)')
ax1.axvline(best_thresh, color='red', ls='--', label=f'Optimal ({best_thresh:.2f})')
ax1.set_xlabel('Threshold'); ax1.set_ylabel('Net Value (Rp)', color='b')
ax2 = ax1.twinx()
ax2.plot(thresholds_range, precs, 'g--', alpha=0.7, label='Precision')
ax2.plot(thresholds_range, recs, 'r--', alpha=0.7, label='Recall')
ax2.set_ylabel('Precision / Recall')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.set_title('Threshold Optimization'); plt.savefig(f'{FIG}/08_threshold_optimization.png'); plt.close()
print("   Saved 08_threshold_optimization.png")

# 9: Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legitimate','Fraud'], yticklabels=['Legitimate','Fraud'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix')
plt.savefig(f'{FIG}/09_confusion_matrix.png'); plt.close()
print("   Saved 09_confusion_matrix.png")

# 9: ROC + PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
ax1.plot(fpr, tpr, 'b-', lw=2, label=f'XGBoost (AUC={final_metrics["auc"]:.4f})')
ax1.plot([0,1],[0,1],'k--',alpha=0.3); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.set_title('ROC Curve'); ax1.legend()
p_arr, r_arr, _ = precision_recall_curve(y_test, y_prob_test)
ax2.plot(r_arr, p_arr, 'g-', lw=2); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('PR Curve'); ax2.axhline(y_test.mean(), color='r', ls='--', alpha=0.5, label=f'Baseline ({y_test.mean():.4f})')
ax2.legend(); plt.tight_layout(); plt.savefig(f'{FIG}/09_roc_pr_curves.png'); plt.close()
print("   Saved 09_roc_pr_curves.png")

# 10: SHAP
import shap
print("   Computing SHAP...")
explainer = shap.TreeExplainer(model)
sample_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test[sample_idx], feature_names=feature_cols, show=False, max_display=15)
plt.title('SHAP Feature Importance'); plt.tight_layout()
plt.savefig(f'{FIG}/10_shap_summary.png'); plt.close()
print("   Saved 10_shap_summary.png")

# 11: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
imp = model.feature_importances_
idx = np.argsort(imp)[-15:]
ax.barh(np.array(feature_cols)[idx], imp[idx], color='#2196F3')
ax.set_xlabel('Importance'); ax.set_title('XGBoost Feature Importance (Top 15)')
plt.tight_layout(); plt.savefig(f'{FIG}/11_feature_importance.png'); plt.close()
print("   Saved 11_feature_importance.png")

# 12: Precision@K chart
k_range = range(10, min(1001, len(y_test)), 10)
p_at_k_vals = []
sorted_idx = np.argsort(y_prob_test)[::-1]
for k in k_range:
    top_k_pred = np.zeros_like(y_test)
    top_k_pred[sorted_idx[:k]] = 1
    p_at_k_vals.append(precision_score(y_test, top_k_pred, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), p_at_k_vals, 'b-', lw=2)
ax.axhline(y_test.mean(), color='r', ls='--', alpha=0.5, label=f'Random baseline ({y_test.mean():.4f})')
ax.set_xlabel('K (top-K transactions flagged)'); ax.set_ylabel('Precision@K')
ax.set_title('Precision@K — Fraud Detection'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{FIG}/12_precision_at_k.png'); plt.close()
print("   Saved 12_precision_at_k.png")

# ===== 11. SAVE RESULTS =====
results = {
    'dataset': 'PaySim 6.3M (full)',
    'split': 'step-based (train≤500, val 501-620, test >620)',
    'features': len(feature_cols),
    'imbalance_method': 'SMOTE',
    'threshold': float(best_thresh),
    'test_metrics': final_metrics,
    'train_auc': float(train_auc),
    'val_auc': float(val_auc),
    'test_auc': float(test_auc),
    'feature_importance_top5': list(np.array(feature_cols)[np.argsort(model.feature_importances_)[-5:][::-1]]),
}
with open('/home/ubuntu/fraudshield/reports/model_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE — {time.time()-t0:.0f}s total")
print(f"{'='*70}")
print(f"Train AUC: {train_auc:.4f}")
print(f"Val AUC:   {val_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Overfit gap (train-test): {train_auc - test_auc:.4f}")
print(f"{'='*70}")
