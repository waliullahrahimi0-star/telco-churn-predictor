import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📞 IBM Telco Customer Churn Predictor")
st.markdown("**Predict if a customer will leave** – built for your coursework")

model = joblib.load('best_churn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 118.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior = st.sidebar.checkbox("Senior Citizen")

data = {
    'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    'SeniorCitizen': 1 if senior else 0, 'Contract': contract, 'InternetService': internet,
    'OnlineSecurity': security, 'PaperlessBilling': paperless, 'PaymentMethod': payment
}
input_df = pd.DataFrame([data])
input_encoded = pd.get_dummies(input_df, drop_first=True)

for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]

if st.button("🔮 Predict Churn", type="primary"):
    prob = model.predict_proba(input_encoded)[0][1]
    prediction = "YES (will churn)" if prob > 0.5 else "NO (will stay)"
    st.subheader(f"Prediction: **{prediction}**")
    st.progress(prob)
    st.write(f"**Churn Probability: {prob:.1%}**")
    
    if prob > 0.7: st.error("🔴 HIGH RISK – Offer discount now!")
    elif prob > 0.4: st.warning("🟡 Medium risk")
    else: st.success("🟢 Low risk – Loyal customer")
