import streamlit as st
import pandas as pd
import pickle

with open("customer_churn_model.pkl","rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["feature_names"]

with open("encoders.pkl","rb") as f:
    encoders = pickle.load(f)
st.title("Telecom Customer Churn Prediction")
st.write("Enter customer details")

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0,1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.number_input(
    "Tenure",
    min_value=0
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No","Yes","No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges"
)

TotalCharges = st.number_input(
    "Total Charges"
)

if st.button("Predict"):

    input_data = {

        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    input_df = pd.DataFrame([input_data])

    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    input_df = input_df[feature_names]

    prediction = loaded_model.predict(input_df)

    pred_prob = loaded_model.predict_proba(input_df)

    result = (
        "Churn"
        if prediction[0] == 1
        else "No Churn"
    )

    st.subheader(f"Prediction: {result}")

    st.write(
        f"Prediction Probability: {pred_prob}"
    )