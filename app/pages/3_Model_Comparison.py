import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.caption("Results from the full 404,800-row dataset, Google Colab, August 2026")

st.subheader("Classification — EMI Eligibility")
clf_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.7897, 0.9439, 0.9626],
    "Macro F1": [0.6385, 0.6512, 0.8756],
    "ROC-AUC": [0.9256, 0.9785, 0.9943],
    "High_Risk Recall": [0.67, 0.03, 0.89],
})
st.dataframe(clf_df, use_container_width=True, hide_index=True)

st.warning(
    "Random Forest has the second-highest accuracy (94.4%) but a High_Risk recall of "
    "just 3% — it almost never catches genuinely risky applicants. XGBoost has both the "
    "highest accuracy (96.3%) and by far the best High_Risk detection (89% recall). "
    "This is why XGBoost — not the accuracy leader alone — was selected for deployment.",
    icon="⚠️"
)

st.bar_chart(clf_df.set_index("Model")[["Accuracy", "Macro F1", "ROC-AUC"]])

st.divider()

st.subheader("Regression — Maximum Safe EMI")
reg_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE (₹)": [4110.18, 889.02, 658.34],
    "MAE (₹)": [2945.86, 279.27, 250.49],
    "R²": [0.7138, 0.9866, 0.9927],
})
st.dataframe(reg_df, use_container_width=True, hide_index=True)

st.success(
    "XGBoost achieves RMSE ₹658 — well under the project's ₹2,000 target. "
    "Linear Regression's poor fit (R²=0.71) confirms the salary-to-EMI relationship "
    "is genuinely non-linear.",
    icon="✅"
)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(reg_df.set_index("Model")["RMSE (₹)"])
with col2:
    st.bar_chart(reg_df.set_index("Model")["R²"])

st.divider()
st.subheader("Deployed Models")
st.markdown("""
| Task | Deployed Model | Why |
|---|---|---|
| Eligibility Classification | **XGBoost** | Highest accuracy + only model that reliably detects High_Risk cases |
| Max EMI Regression | **XGBoost** | Lowest RMSE, highest R², beats target by 3× |
""")
