import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('body_fat_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('scaler.sav', 'rb') as f:
    scaler = pickle.load(f)

# Title and instructions
st.set_page_config(page_title="Body Fat Predictor", layout="centered")
st.title("💪 Body Fat Percentage Predictor")
st.markdown("Enter your daily fitness metrics to estimate your body fat %.")

# Input fields
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
calories = st.number_input("Calories Consumed", min_value=500, max_value=6000, value=2500, step=50)
protein = st.number_input("Protein Intake (g)", min_value=0, max_value=300, value=120, step=5)
sleep = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
steps = st.number_input("Steps Walked", min_value=0, max_value=30000, value=8000, step=500)

# Format input for prediction
input_data = np.array([[weight, calories, protein, sleep, steps]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Body Fat Percentage: **{prediction:.2f}%**")
