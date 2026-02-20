import streamlit as st
import joblib
import numpy as np

# Load updated model and encoder
model = joblib.load("insurance_model_updated.pkl")  # model trained with 4 extra features + log
oe = joblib.load("region_encoder.pkl")

st.title("Insurance Cost Predictor")

# User inputs
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
    # Encode categorical variables
    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    # OneHotEncode region
    region_encoded = oe.transform([[region]])  # shape (1,4)

    # Compute interaction features
    children_bmi = children * bmi
    children_age = children * age
    smoker_bmi = smoker_val * bmi
    smoker_age = smoker_val * age

    # Combine all features in correct order
    X_user = np.concatenate([
        region_encoded, 
        [[age, sex_val, bmi, children, smoker_val,
          children_bmi, children_age, smoker_bmi, smoker_age]]
    ], axis=1)

    # Predict using log-trained model
    prediction_log = model.predict(X_user)
    prediction = np.expm1(prediction_log)  # inverse log1p

    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")