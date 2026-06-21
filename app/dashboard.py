"""
FraudShield-Lite Dashboard — Interactive Fraud Detection UI

Features:
- Transaction fraud prediction (interactive input)
- Model performance metrics (ROC, PR, confusion matrix)
- SHAP explainability
- Cost analysis
- Baseline comparison

Run: streamlit run app/dashboard.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
@st.cache_data
def _load_dashboard_config():
    cfg_path = Path("config/config.yaml")
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield-Lite 🛡️",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model + config
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model.json")
    return model

@st.cache_data
def load_reports():
    with open("reports/baseline_comparison.json") as f:
        baseline = json.load(f)
    with open("reports/distribution_shift_analysis.json") as f:
        shift = json.load(f)
    with open("metrics.json") as f:
        metrics = json.load(f)
    return baseline, shift, metrics

@st.cache_data
def load_features():
    if os.path.exists("data/features.csv"):
        return pd.read_csv("data/features.csv")
    # Auto-generate demo data for Streamlit Cloud (no data files needed)
    return _generate_demo_data()

def _generate_demo_data():
    """Generate realistic demo data matching the 36-feature schema."""
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "step": np.random.randint(1, 744, n),
        "amount": np.random.exponential(500000, n),
        "oldbalanceOrg": np.random.exponential(1000000, n),
        "newbalanceOrig": np.random.exponential(800000, n),
        "oldbalanceDest": np.random.exponential(500000, n),
        "newbalanceDest": np.random.exponential(700000, n),
    })
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_to_dest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["orig_balance_drained"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["dest_balance_received"] = (df["newbalanceDest"] > df["oldbalanceDest"]).astype(int)
    df["step_tx_count"] = np.random.randint(50, 500, n)
    df["step_avg_amount"] = df["amount"] * np.random.uniform(0.8, 1.2, n)
    df["step_total_amount"] = df["step_tx_count"] * df["step_avg_amount"]
    df["orig_tx_cumcount"] = np.random.randint(1, 100, n)
    df["orig_amount_cumsum"] = df["amount"] * df["orig_tx_cumcount"]
    df["orig_amount_cummean"] = df["amount"]
    df["dest_tx_cumcount"] = np.random.randint(1, 100, n)
    df["dest_amount_cumsum"] = df["amount"] * np.random.uniform(0.5, 2, n)
    df["amount_dev_from_orig_mean"] = np.random.normal(0, 100000, n)
    df["amount_ratio_to_orig_mean"] = np.random.lognormal(0, 0.5, n)
    # Velocity features
    df["tx_count_24h"] = np.random.randint(0, 10, n)
    df["tx_count_7d"] = np.random.randint(0, 30, n)
    df["amt_sum_24h"] = np.random.exponential(1000000, n)
    df["amt_sum_7d"] = np.random.exponential(5000000, n)
    df["amt_mean_24h"] = np.random.exponential(300000, n)
    df["amt_mean_7d"] = np.random.exponential(400000, n)
    df["avg_time_between_tx"] = np.random.exponential(24, n)
    # Type encoding
    types = np.random.choice(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"], n, p=[0.35, 0.25, 0.25, 0.1, 0.05])
    for t in ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]:
        df[f"tx_type_{t}"] = (types == t).astype(int)
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = ((df["step"] % 24) >= 22) | ((df["step"] % 24) <= 5)
    df["is_night"] = df["is_night"].astype(int)
    # Generate synthetic fraud labels (0.1% rate)
    fraud_prob = (df["amount_to_orig_ratio"] * 0.3 + df["orig_balance_drained"] * 0.4 + df["is_night"] * 0.1 + np.random.exponential(0.01, n))
    df["isFraud"] = (fraud_prob > np.percentile(fraud_prob, 99.9)).astype(int)
    df["isFlaggedFraud"] = 0
    return df

model = load_model()
baseline, shift, metrics = load_reports()
features_df = load_features()
dashboard_config = _load_dashboard_config()
DEFAULT_THRESHOLD = dashboard_config.get("thresholds", {}).get("default", 0.45)

# Feature columns
FEATURE_COLS = [
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
    "tx_count_24h", "tx_count_7d",
    "amt_sum_24h", "amt_sum_7d",
    "amt_mean_24h", "amt_mean_7d",
    "avg_time_between_tx",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ FraudShield-Lite")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Predict", "📊 Model Performance",
     "💡 Explainability", "💰 Cost Analysis", "📈 Walk-Forward CV"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model:** v1.0 (XGBoost)  \n"
    f"**Features:** {len(FEATURE_COLS)}  \n"
    f"**Dataset:** PaySim 500K rows  \n"
    f"**Walk-forward F1:** {metrics.get('f1', 'N/A')}"
)

# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def page_overview():
    st.title("🛡️ FraudShield-Lite — Overview")
    st.markdown("Anti-fraud & AML scoring engine for e-wallet systems.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.markdown("---")

    # Baseline comparison
    st.subheader("Model Comparison")
    df_bl = pd.DataFrame(baseline)
    fig = go.Figure()
    for _, row in df_bl.iterrows():
        fig.add_trace(go.Bar(
            name=row["model"].split("(")[0].strip(),
            x=["ROC-AUC", "F1"],
            y=[row["roc_auc"], row["f1_fraud"]],
            text=[f"{row['roc_auc']:.3f}", f"{row['f1_fraud']:.3f}"],
            textposition="auto",
        ))
    fig.update_layout(barmode="group", height=400, yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Cost comparison
    st.subheader("💰 Net Value Comparison")
    fig2 = go.Figure()
    colors = ["#ef4444", "#f59e0b", "#22c55e", "#6366f1"]
    for i, (_, row) in enumerate(df_bl.iterrows()):
        fig2.add_trace(go.Bar(
            name=row["model"].split("(")[0].strip(),
            x=["Net Value (Rp)"],
            y=[row["net_value_rp"]],
            marker_color=colors[i],
            text=[f"Rp {row['net_value_rp']:,.0f}"],
            textposition="auto",
        ))
    fig2.update_layout(height=350, yaxis_title="Rp")
    st.plotly_chart(fig2, use_container_width=True)

    # Fraud distribution
    if features_df is not None:
        st.subheader("📊 Fraud Distribution")
        col1, col2 = st.columns(2)
        with col1:
            type_fraud = features_df.groupby(["tx_type_CASH_OUT", "tx_type_TRANSFER", "tx_type_PAYMENT"]).agg(
                total=("isFraud", "count"),
                fraud=("isFraud", "sum")
            ).reset_index()
            # Simplify: just use raw type
            st.bar_chart(features_df["isFraud"].value_counts())
            st.caption("Fraud vs Legit")
        with col2:
            fraud_by_type = pd.DataFrame({
                "Type": ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
                "Fraud": [
                    features_df[features_df["tx_type_CASH_OUT"] == 1]["isFraud"].sum(),
                    features_df[features_df["tx_type_TRANSFER"] == 1]["isFraud"].sum(),
                    features_df[features_df["tx_type_PAYMENT"] == 1]["isFraud"].sum(),
                    features_df[features_df["tx_type_CASH_IN"] == 1]["isFraud"].sum(),
                    features_df[features_df["tx_type_DEBIT"] == 1]["isFraud"].sum(),
                ]
            })
            fig3 = px.bar(fraud_by_type, x="Type", y="Fraud", color="Type",
                         title="Fraud Cases by Transaction Type")
            st.plotly_chart(fig3, use_container_width=True)


def page_predict():
    st.title("🔍 Fraud Prediction")
    st.markdown("Input a transaction and get a real-time fraud prediction.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        step = st.slider("Step (hour)", 1, 744, 300)
        amount = st.number_input("Amount (Rp)", min_value=1.0, value=500000.0, step=10000.0)
        tx_type = st.selectbox("Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])

        st.subheader("Sender Balance")
        oldbalanceOrg = st.number_input("Old Balance", min_value=0.0, value=1000000.0)
        newbalanceOrig = st.number_input("New Balance", min_value=0.0, value=500000.0)

    with col2:
        st.subheader("Receiver Balance")
        oldbalanceDest = st.number_input("Dest Old Balance", min_value=0.0, value=500000.0)
        newbalanceDest = st.number_input("Dest New Balance", min_value=0.0, value=1000000.0)

        st.subheader("Velocity (optional)")
        step_tx_count = st.number_input("Transactions in this step", min_value=1, value=100)
        orig_tx_cumcount = st.number_input("Sender total tx count", min_value=1, value=50)
        dest_tx_cumcount = st.number_input("Receiver total tx count", min_value=1, value=50)
        tx_count_24h = st.number_input("Sender tx in last 24h", min_value=0, value=0)
        tx_count_7d = st.number_input("Sender tx in last 7d", min_value=0, value=0)
        st.session_state["tx_count_24h"] = tx_count_24h
        st.session_state["tx_count_7d"] = tx_count_7d
        st.session_state["amt_sum_24h"] = float(amount * tx_count_24h)
        st.session_state["amt_sum_7d"] = float(amount * tx_count_7d)
        st.session_state["amt_mean_24h"] = float(amount) if tx_count_24h > 0 else 0.0
        st.session_state["amt_mean_7d"] = float(amount) if tx_count_7d > 0 else 0.0
        st.session_state["avg_time_between_tx"] = 24.0 if tx_count_24h > 1 else 0.0

    if st.button("🚀 Predict Fraud", type="primary", use_container_width=True):
        # Build feature vector
        balance_diff_orig = oldbalanceOrg - newbalanceOrig
        balance_diff_dest = oldbalanceDest - newbalanceDest
        amount_to_orig_ratio = amount / (oldbalanceOrg + 1)
        amount_to_dest_ratio = amount / (oldbalanceDest + 1)
        orig_balance_drained = int(newbalanceOrig == 0 and oldbalanceOrg > 0)
        dest_balance_received = int(newbalanceDest > oldbalanceDest)

        type_map = {"CASH_IN": "tx_type_CASH_IN", "CASH_OUT": "tx_type_CASH_OUT",
                     "DEBIT": "tx_type_DEBIT", "PAYMENT": "tx_type_PAYMENT",
                     "TRANSFER": "tx_type_TRANSFER"}

        features = {}
        for col in FEATURE_COLS:
            features[col] = 0.0

        features["step"] = step
        features["amount"] = amount
        features["oldbalanceOrg"] = oldbalanceOrg
        features["newbalanceOrig"] = newbalanceOrig
        features["oldbalanceDest"] = oldbalanceDest
        features["newbalanceDest"] = newbalanceDest
        features["balance_diff_orig"] = balance_diff_orig
        features["balance_diff_dest"] = balance_diff_dest
        features["amount_to_orig_ratio"] = amount_to_orig_ratio
        features["amount_to_dest_ratio"] = amount_to_dest_ratio
        features["orig_balance_drained"] = orig_balance_drained
        features["dest_balance_received"] = dest_balance_received
        features["step_tx_count"] = step_tx_count
        features["step_avg_amount"] = amount  # approximate
        features["step_total_amount"] = amount * step_tx_count
        features["orig_tx_cumcount"] = orig_tx_cumcount
        features["orig_amount_cumsum"] = amount * orig_tx_cumcount
        features["orig_amount_cummean"] = amount
        features["dest_tx_cumcount"] = dest_tx_cumcount
        features["dest_amount_cumsum"] = amount
        features["amount_dev_from_orig_mean"] = 0
        features["amount_ratio_to_orig_mean"] = 1.0
        features[type_map[tx_type]] = 1.0
        features["hour_of_day"] = step % 24
        features["is_night"] = int((step % 24) >= 22 or (step % 24) <= 5)
        features["tx_count_24h"] = st.session_state.get("tx_count_24h", 0)
        features["tx_count_7d"] = st.session_state.get("tx_count_7d", 0)
        features["amt_sum_24h"] = st.session_state.get("amt_sum_24h", 0.0)
        features["amt_sum_7d"] = st.session_state.get("amt_sum_7d", 0.0)
        features["amt_mean_24h"] = st.session_state.get("amt_mean_24h", 0.0)
        features["amt_mean_7d"] = st.session_state.get("amt_mean_7d", 0.0)
        features["avg_time_between_tx"] = st.session_state.get("avg_time_between_tx", 0.0)

        x = np.array([[features[col] for col in FEATURE_COLS]])
        proba = float(model.predict_proba(x)[0, 1])
        pred = 1 if proba >= DEFAULT_THRESHOLD else 0

        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        if pred == 1:
            r1.error(f"🚨 **FRAUD** (probability: {proba:.2%})")
            if proba >= 0.65:
                r2.error("**Decision: BLOCK**")
            else:
                r2.warning("**Decision: REVIEW**")
        else:
            r1.success(f"✅ **LEGIT** (probability: {proba:.2%})")
            r2.success("**Decision: APPROVE**")

        r3.metric("Fraud Probability", f"{proba:.4f}")

        # Risk factors
        st.subheader("Risk Factors")
        risks = []
        if balance_diff_orig > amount * 0.9:
            risks.append("⚠️ **Balance nearly drained** — sender balance dropped significantly")
        if amount_to_orig_ratio > 0.8:
            risks.append("⚠️ **High amount-to-balance ratio** — transaction is large relative to balance")
        if orig_balance_drained:
            risks.append("🔴 **Balance drained to zero** — classic fraud pattern")
        if step % 24 >= 22 or step % 24 <= 5:
            risks.append("🌙 **Night transaction** — higher risk period")
        if amount > 10_000_000:
            risks.append("💰 **High-value transaction** — above Rp 10M threshold")

        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.info("No significant risk factors detected.")


def page_performance():
    st.title("📊 Model Performance")


    feature_cols = [c for c in features_df.columns if c not in ["isFraud", "isFlaggedFraud"]]
    test = features_df[features_df["step"] > 600]
    X_test = test[feature_cols]
    y_test = test["isFraud"]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"],
            colorscale="Blues", text=cm, texttemplate="%{text}",
        ))
        fig.update_layout(height=350, xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.4f}"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))
        fig2.update_layout(height=350, xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig2, use_container_width=True)

    # PR Curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=recall, y=precision, name=f"PR-AUC = {pr_auc:.4f}"))
    fig3.update_layout(height=350, xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig3, use_container_width=True)

    # Metrics table
    st.subheader("Detailed Metrics")
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_cr = pd.DataFrame(cr).T
    st.dataframe(df_cr.style.format("{:.4f}"), use_container_width=True)


def page_explainability():
    st.title("💡 SHAP Explainability")


    try:
        import shap
    except ImportError:
        st.error("Install shap: `pip install shap`")
        return

    feature_cols = [c for c in features_df.columns if c not in ["isFraud", "isFlaggedFraud"]]
    sample = features_df[feature_cols].sample(n=min(1000, len(features_df)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    st.subheader("Feature Importance (SHAP)")
    # Mean absolute SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    df_shap = pd.DataFrame({"Feature": feature_cols, "Mean |SHAP|": mean_shap})
    df_shap = df_shap.sort_values("Mean |SHAP|", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_shap["Mean |SHAP|"], y=df_shap["Feature"], orientation="h",
        marker_color="#6366f1",
    ))
    fig.update_layout(height=600, xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Higher SHAP values = more impact on fraud prediction. "
            "balance_diff_orig and amount features dominate.")


def page_cost():
    st.title("💰 Cost Analysis")

    st.subheader("Cost Matrix")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Cost (FN)", "Rp 500,000")
    c2.metric("FP Cost (review)", "Rp 15,000")
    c3.metric("Cost Ratio", "33:1")

    st.markdown("---")

    # Net value by model
    st.subheader("Net Value by Model")
    df_bl = pd.DataFrame(baseline)
    fig = px.bar(df_bl, x="model", y="net_value_rp",
                 color="model", title="Monthly Net Value (Rp)",
                 color_discrete_map={
                     "Dummy (most_frequent)": "#ef4444",
                     "Logistic Regression (balanced)": "#f59e0b",
                     "XGBoost (scale_pos_weight)": "#22c55e",
                 })
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    st.subheader("Annual Cost Breakdown (XGBoost)")
    costs = pd.DataFrame({
        "Category": ["Fraud Prevented", "FP Review Cost", "Model Inference", "Team Cost", "Net Savings"],
        "Annual (Rp B)": [84.79, -1.94, -0.03, -0.60, 82.22],
        "Type": ["Savings", "Cost", "Cost", "Cost", "Savings"],
    })
    fig2 = px.bar(costs, x="Category", y="Annual (Rp B)", color="Type",
                  color_discrete_map={"Savings": "#22c55e", "Cost": "#ef4444"},
                  title="Annual Financial Impact")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("**ROI: 3,300%** — For every Rp 1 invested, save Rp 33 in fraud losses.")


def page_walkforward():
    st.title("📈 Walk-Forward CV Results")

    st.markdown("""
    **Method**: 3-fold expanding window, time-respecting split.
    Trains on past → predicts future (matches production).
    """)

    # Fold results
    folds = shift.get("walk_forward", [])
    if folds:
        df_wf = pd.DataFrame(folds)
        st.subheader("Per-Fold Metrics")
        st.dataframe(df_wf[["fold", "train_steps", "test_steps", "roc_auc", "pr_auc",
                            "precision", "recall", "f1"]].style.format({
            "roc_auc": "{:.4f}", "pr_auc": "{:.4f}",
            "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"
        }), use_container_width=True)

        # Trend visualization
        st.subheader("Metrics Across Folds")
        fig = go.Figure()
        for metric in ["roc_auc", "precision", "recall", "f1"]:
            fig.add_trace(go.Scatter(
                x=df_wf["fold"], y=df_wf[metric],
                name=metric.upper().replace("_", "-"),
                mode="lines+markers",
            ))
        fig.update_layout(height=400, xaxis_title="Fold", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution shift explanation
    st.subheader("🔄 Distribution Shift")
    shift_rows = []
    for f in folds:
        shift_ratio = f.get("test_fraud_rate", 0) / max(f.get("train_fraud_rate", 0.001), 0.0001)
        shift_rows.append({
            "Fold": f["fold"],
            "Train Fraud Rate": f"{f.get('train_fraud_rate', 0)*100:.3f}%",
            "Test Fraud Rate": f"{f.get('test_fraud_rate', 0)*100:.3f}%",
            "Shift": f"{shift_ratio:.1f}×",
        })
    if shift_rows:
        df_shift = pd.DataFrame(shift_rows)
        # Build a markdown table string
        header = "| Fold | Train Fraud Rate | Test Fraud Rate | Shift |\n"
        header += "|------|:----------------:|:---------------:|:-----:|\n"
        body_lines = []
        for _, r in df_shift.iterrows():
            body_lines.append(f"| {int(r['Fold'])} | {r['Train Fraud Rate']} | {r['Test Fraud Rate']} | {r['Shift']} |")
        body = "\n".join(body_lines)
        st.markdown(header + body)

    st.markdown("""
    **Key insight**: Fraud rate increases over time. Fold 3 (latest period) has 14× more
    fraud than training. This is realistic — fraud patterns evolve.
    """)

    mean = shift.get("walk_forward_mean", {})
    st.success(
        f"**Walk-Forward Mean**: ROC-AUC {mean.get('roc_auc', 'N/A')}, "
        f"Precision {mean.get('precision', 'N/A')}, "
        f"Recall {mean.get('recall', 'N/A')}, "
        f"F1 {mean.get('f1', 'N/A')}"
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Overview": page_overview,
    "🔍 Predict": page_predict,
    "📊 Model Performance": page_performance,
    "💡 Explainability": page_explainability,
    "💰 Cost Analysis": page_cost,
    "📈 Walk-Forward CV": page_walkforward,
}

PAGES[page]()

# Footer
st.markdown("---")
st.caption("FraudShield-Lite | Built with XGBoost + Streamlit | Portfolio by Hadijayyy")
