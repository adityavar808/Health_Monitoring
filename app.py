import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

model = joblib.load('health_monitoring.pkl')
st.title("Health Diagnosis Predictor")
st.image("health.jpg")
Heart_Rate_bpm = st.slider("Heart Rate (bpm) ❤️", max_value=300, value=100)
Body_Temperature_C = st.slider("Body Temperature (°C) 🔥", max_value=150, value=97)
Blood_Pressure_mmHg = st.slider("Blood Preesure (mmHg) 🩸", max_value=300, value=120)
Oxygen_Saturation = st.slider("Oxygen Saturation (%) 🍃", max_value=100, value=95)

input_data = pd.DataFrame({
    'Heart_Rate_bpm' : [Heart_Rate_bpm],
    'Body_Temperature_C' : [Body_Temperature_C],
    'Blood_Pressure_mmHg' :[Blood_Pressure_mmHg],
    'Oxygen_Saturation_%' : [Oxygen_Saturation]
})


if st.button('Predict'):
    with st.spinner('Wait for it...'):
        time.sleep(5)

    st.snow()
    health = model.predict(input_data)
    print(health)
    st.write("Health Diagnosis:",health)
    st.info('Thank you for using the Health Diagnosis Predictor!')
    