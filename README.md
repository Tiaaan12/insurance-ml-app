# Geospatial-flood-risk-prediction

Project Overview

This application predicts medical insurance premiums based on individual health and demographic factors. It uses a regression model trained on the Medical Cost Personal Dataset, incorporating feature engineering to capture relationships between variables such as smoking habits and BMI.


Technical Stack

• Framework: Streamlit

• Model: Scikit-Learn (Gradient Boosting/Regression)

• Preprocessing: Ordinal Encoding for categorical regional data

• Math Operations: NumPy (Logarithmic transformation handling)


Project Structure

• insurance_app.py: Main Streamlit application and UI logic.

• insurance_model_updated.pkl: Trained model file.

• region_encoder.pkl: Pickled encoder for regional categorical data.

• requirements.txt: List of dependencies for deployment.


Key Features

• Feature Interaction: The app calculates custom interaction terms (e.g., Smoker × BMI) in real-time to improve prediction accuracy.

• Log Transformation: Uses np.expm1 to reverse log-transformed targets, ensuring accurate dollar-value outputs.

• Interactive UI: Simple input fields for age, BMI, and lifestyle choices.


🔗 Live Demo: https://insurance-ml-app.streamlit.app/
