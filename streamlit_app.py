import streamlit as st
import pandas as pd
import pickle
import numpy as np

# App Title
st.title("Income Prediction App")

# Load the trained model
with open('classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



age = st.number_input("Age", min_value=0, max_value=100, value=30)
education = st.selectbox("Education Level", [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th',
    '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm',
    'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
marital_status = st.selectbox("Married (Civ-Spouse)?", ["Yes", "No"])
occupation_exec_mgr = st.selectbox("Occupation: Exec-Managerial?", ["Yes", "No"])

# Create DataFrame from inputs
input_df = pd.DataFrame([{
    'age': age,
    'education': education,
    'sex': sex,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'marital_Married-civ-spouse': marital_status,
    'occupation_Exec-managerial': occupation_exec_mgr
}])

# Encoding categorical variables
education_order = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4,
    '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9,
    'Assoc-voc': 10, 'Assoc-acdm': 11, 'Bachelors': 12,
    'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
input_df['education'] = input_df['education'].map(education_order)
input_df['sex'] = input_df['sex'].map({'Male': 0, 'Female': 1})
input_df['marital_Married-civ-spouse'] = input_df['marital_Married-civ-spouse'].map({'Yes': 1, 'No': 0})
input_df['occupation_Exec-managerial'] = input_df['occupation_Exec-managerial'].map({'Yes': 1, 'No': 0})

# Normalize numeric features
numeric_cols = [
    'age', 'education', 'capital-gain', 'capital-loss', 'hours-per-week',
    'marital_Married-civ-spouse', 'occupation_Exec-managerial'
]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict button
if st.button("Predict Income Class"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income Class: {label}")
