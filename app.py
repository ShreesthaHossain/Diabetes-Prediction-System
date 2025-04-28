import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model from the 'notebooks' directory
model = joblib.load('notebooks/diabetes_model.pkl')

# Load the dataset (optional, if you want to show data or use it in any part of the app)
data = pd.read_csv('data/diabetes.csv')

# Add custom CSS to style the button
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        height: 50px;
        width: 100%;
    }

    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

# Add custom header with style
st.markdown('<h1 style="color: darkblue; text-align: center;">Diabetes Prediction System</h1>', unsafe_allow_html=True)

# Move input fields to sidebar with unique keys
pregnancies = st.sidebar.number_input('Number of Pregnancies (count)', min_value=0, key="pregnancies")
glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, key="glucose")
blood_pressure = st.sidebar.text_input('Blood Pressure (mmHg)', value='0/0', key="blood_pressure")  # Single input box for BP
skin_thickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, key="skin_thickness")
insulin = st.sidebar.number_input('Insulin Level (mu U/mL)', min_value=0, key="insulin")
bmi = st.sidebar.number_input('BMI (kg/m²)', min_value=0.0, key="bmi")
diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function (ratio)', min_value=0.0, key="diabetes_pedigree_function")
age = st.sidebar.number_input('Age (years)', min_value=0, key="age")

# Prediction button with icon
if st.button('Predict 💡'):
    # Validate blood pressure input and split into systolic/diastolic
    try:
        systolic, diastolic = map(int, blood_pressure.split('/'))  # Blood pressure input as 120/80
        combined_bp = (systolic + diastolic) / 2  # Averaging them for simplicity (modify this as per your model's logic)
    except ValueError:
        st.error("Please enter blood pressure in the format: 'Systolic/Diastolic', e.g., 120/80.")
        combined_bp = 0  # Default if the input is invalid

    # Prepare the input data in the same format the model expects (1D array)
    input_data = np.array([pregnancies, glucose, combined_bp, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
    
    # Make the prediction using the model
    prediction = model.predict(input_data)
    
    # Display prediction results in stylish boxes
    if prediction == 1:
        st.markdown('<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; text-align: center;">The model predicts: <b>Diabetic</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: green; color: white; padding: 10px; border-radius: 5px; text-align: center;">The model predicts: <b>Not Diabetic</b></div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        left: 60%;
        transform: translateX(-50%);
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
        Created with ❤️ by DiaDetect
    </div>
    """, 
    unsafe_allow_html=True
)