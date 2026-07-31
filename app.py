import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Startup Success Predictor")

model = joblib.load("startup_model.pkl")
scaler = joblib.load("Scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("AI Startup Success Predictor")

funding = st.number_input("Funding Amount", min_value=0)

experience = st.number_input("Founder Experience", min_value=0)

team = st.number_input("Team Size", min_value=1)

industry = st.selectbox(
    "Industry",
    ["FinTech","SaaS","E-commerce","HealthTech","AI"]
)

market = st.selectbox(
    "Market Size",
    ["Small","Medium","Large"]
)

competition = st.selectbox(
    "Competition Level",
    ["Low","Medium","High"]
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Industry":[industry],
        "Founder_Experience":[experience],
        "Team_Size":[team],
        "Funding_USD":[funding],
        "Market_Size":[market],
        "AI_Adoption":["High"],
        "Customer_Growth":[80],
        "Burn_Rate":[25],
        "Revenue":[500000],
        "Investor_Support":["Yes"],
        "Product_Innovation":[8],
        "Competition_Level":[competition],
        "Funding_per_Employee":[funding/team]
    })

    # Same encoding as training
    input_df = pd.get_dummies(input_df)

    # Same columns as training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    confidence = max(model.predict_proba(input_scaled)[0]) * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(" Successful Startup")
    else:
        st.error(" Not Successful Startup")

    st.write(f"### Probability of Success : {probability*100:.2f}%")

    st.write(f"### Confidence Score : {confidence:.2f}%")