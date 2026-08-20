import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils import (
    predict_eligibility, GENDER_OPTIONS, MARITAL_OPTIONS, EDUCATION_OPTIONS,
    EMPLOYMENT_OPTIONS, COMPANY_OPTIONS, HOUSE_OPTIONS, EXISTING_LOAN_OPTIONS,
    SCENARIO_OPTIONS
)

st.set_page_config(page_title="Eligibility Prediction", page_icon="✅", layout="wide")
st.title("✅ EMI Eligibility Prediction")
st.caption("Powered by XGBoost — 96.3% accuracy, 89% High-Risk detection recall")

with st.form("eligibility_form"):
    st.subheader("Personal & Demographic")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 75, 30)
    gender = c2.selectbox("Gender", GENDER_OPTIONS)
    marital_status = c3.selectbox("Marital Status", MARITAL_OPTIONS)
    education = c1.selectbox("Education", EDUCATION_OPTIONS)
    family_size = c2.number_input("Family Size", 1, 15, 3)
    dependents = c3.number_input("Dependents", 0, 10, 1)

    st.subheader("Employment & Income")
    c1, c2, c3 = st.columns(3)
    monthly_salary = c1.number_input("Monthly Salary (₹)", 0, 2_000_000, 50000, step=1000)
    employment_type = c2.selectbox("Employment Type", EMPLOYMENT_OPTIONS)
    years_of_employment = c3.number_input("Years of Employment", 0, 45, 5)
    company_type = c1.selectbox("Company Type", COMPANY_OPTIONS)

    st.subheader("Housing & Expenses")
    c1, c2, c3 = st.columns(3)
    house_type = c1.selectbox("House Type", HOUSE_OPTIONS)
    monthly_rent = c2.number_input("Monthly Rent (₹)", 0, 500_000, 10000, step=500)
    school_fees = c3.number_input("School Fees (₹/month)", 0, 200_000, 0, step=500)
    college_fees = c1.number_input("College Fees (₹/month)", 0, 200_000, 0, step=500)
    travel_expenses = c2.number_input("Travel Expenses (₹/month)", 0, 100_000, 3000, step=500)
    groceries_utilities = c3.number_input("Groceries & Utilities (₹/month)", 0, 200_000, 8000, step=500)
    other_monthly_expenses = c1.number_input("Other Monthly Expenses (₹)", 0, 200_000, 2000, step=500)

    st.subheader("Financial Status & Credit History")
    c1, c2, c3 = st.columns(3)
    existing_loans = c1.selectbox("Existing Loans", EXISTING_LOAN_OPTIONS)
    current_emi_amount = c2.number_input("Current EMI Amount (₹)", 0, 500_000, 0, step=500)
    credit_score = c3.number_input("Credit Score", 300, 900, 700)
    bank_balance = c1.number_input("Bank Balance (₹)", 0, 10_000_000, 50000, step=1000)
    emergency_fund = c2.number_input("Emergency Fund (₹)", 0, 5_000_000, 20000, step=1000)

    st.subheader("Loan Application Details")
    c1, c2, c3 = st.columns(3)
    emi_scenario = c1.selectbox("EMI Scenario", SCENARIO_OPTIONS)
    requested_amount = c2.number_input("Requested Loan Amount (₹)", 0, 20_000_000, 200000, step=5000)
    requested_tenure = c3.number_input("Requested Tenure (months)", 1, 360, 24)

    submitted = st.form_submit_button("Predict Eligibility", type="primary", use_container_width=True)

if submitted:
    raw_input = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=monthly_salary, employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
        dependents=dependents, school_fees=school_fees, college_fees=college_fees,
        travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
        other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
        current_emi_amount=current_emi_amount, credit_score=credit_score,
        bank_balance=bank_balance, emergency_fund=emergency_fund,
        emi_scenario=emi_scenario, requested_amount=requested_amount,
        requested_tenure=requested_tenure,
    )

    label, proba = predict_eligibility(raw_input)

    st.divider()
    if label == "Eligible":
        st.success(f"### Result: {label} ✅")
    elif label == "High_Risk":
        st.warning(f"### Result: {label} ⚠️")
    else:
        st.error(f"### Result: {label} ❌")

    st.subheader("Prediction Confidence")
    cols = st.columns(len(proba))
    for col, (cls, p) in zip(cols, proba.items()):
        col.metric(cls, f"{p*100:.1f}%")
