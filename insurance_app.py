import streamlit as st
import joblib
import numpy as np

model = joblib.load("insurance_model.pkl")
oe = joblib.load("region_encoder.pkl")

st.title("Insurance Cost Predictor")

age = st.number_input("Age", 18, 100, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Children", 0, 10, 0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict"):
    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    region_encoded = oe.transform([[region]])

    X_user = np.array([[age, sex_val, bmi, children, smoker_val]])
    X_user = np.concatenate([X_user, region_encoded], axis=1)

    prediction = model.predict(X_user)

    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")