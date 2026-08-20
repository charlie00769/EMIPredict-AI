import streamlit as st

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")

st.title("💳 EMIPredict AI")
st.subheader("Intelligent EMI Eligibility & Affordability Assessment")

st.markdown("""
This tool uses machine learning models trained on 404,800 customer records to:

- **Predict EMI eligibility** — Eligible / High Risk / Not Eligible
- **Estimate the maximum safe monthly EMI** a customer can afford

Both predictions are powered by **XGBoost**, selected after comparing 3 classification
and 3 regression models on accuracy, F1-score, and RMSE.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Classification Accuracy", "96.3%")
    st.caption("XGBoost, full 404,800-row test set")
with col2:
    st.metric("High-Risk Detection (Recall)", "89%")
    st.caption("vs. 3% for Random Forest — the key reason XGBoost was chosen")
with col3:
    st.metric("Regression R²", "0.993")
    st.caption("RMSE ₹658 — target was < ₹2,000")

st.divider()

st.markdown("""
### How to use this app

1. Go to **Eligibility Prediction** in the sidebar to check if a customer qualifies for a loan
2. Go to **Max EMI Prediction** to estimate the maximum safe monthly EMI amount
3. Go to **Model Comparison** to see how all 6 trained models performed

Use the sidebar on the left to navigate between pages.
""")

st.info(
    "Note: emi_eligibility in the training data is imbalanced (77% Not_Eligible, "
    "18% Eligible, 4% High_Risk). XGBoost was selected specifically because it was "
    "the only model that reliably detects High_Risk cases — see the Model Comparison "
    "page for details.",
    icon="ℹ️"
)
