"""Week 2 — Baseline Modeling Script"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/fraudshield')

from src.features import create_balance_features, create_behavioral_features, create_structuring_features
from src.models import time_based_split, calculate_business_cost, train_baseline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
FIG_DIR = '/home/ubuntu/fraudshield/reports/figures'

# ===== LOAD & FEATURES =====
print("Loading data (100K rows)...")
df = pd.read_csv('/home/ubuntu/fraudshield/data/raw/paysim.csv', nrows=100000)
df = create_balance_features(df)
df = create_behavioral_features(df)
df = create_structuring_features(df)
df['tx_count_total'] = df.groupby('nameOrig')['step'].transform('count')

df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
drop_cols = ['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest', 'type']
feature_cols = [c for c in df_model.columns if c not in drop_cols]

train, val, test = time_based_split(df_model, time_col='step', test_size=0.15, val_size=0.1)

X_train = train[feature_cols].fillna(0).values
y_train = train['isFraud'].values
X_test = test[feature_cols].fillna(0).values
y_test = test['isFraud'].values

# ===== BASELINES =====
print("\nTraining baselines...")
baseline_results = train_baseline(X_train, y_train, X_test, y_test)

for name, metrics in baseline_results.items():
    print(f"  {name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

# ===== ROC CURVES =====
fig, ax = plt.subplots(figsize=(8, 6))
models = {
    'Dummy': (DummyClassifier(strategy='most_frequent', random_state=42), 'gray'),
    'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42), '#2196F3'),
}
for label, (model, color) in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{label} (AUC={auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Baseline Model ROC Curves', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig(f'{FIG_DIR}/07_baseline_roc.png')
plt.close()
print("Saved 07_baseline_roc.png")

# ===== BUSINESS COST =====
print("\n=== BUSINESS COST ANALYSIS ===")
for name in ['dummy', 'logistic_regression']:
    if name == 'dummy':
        m = DummyClassifier(strategy='most_frequent', random_state=42)
    else:
        m = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    cost = calculate_business_cost(y_test, y_pred)
    print(f"  {name}: net_value=Rp {cost['net_value_rp']:,.0f}, "
          f"precision={cost['precision']:.4f}, recall={cost['recall']:.4f}")

print("\n✅ Baseline modeling complete!")
