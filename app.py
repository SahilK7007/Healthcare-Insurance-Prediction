
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('healthcare_model.pkl')

#st.set_page_config(page_title = "Healthcare Premium Predictor")
st.set_page_config(
    page_title="Healthcare Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Health Care Premium Predictor")
st.write("Enter Your Details Below ot estimate your annul insurance premium")

col1,col2 = st.columns(2)

with col1:
    age = st.slider("Select your Age",18,100,30)
    bmi = st.number_input("Enter your BMI",min_value = 10.0,max_value = 50.0,value = 25.0)

with col2:
    smoker_choice = st.selectbox("Are You a Smoker ?",["No","Yes"])

if st.button("Calculate Premium"):
    smoker = 1 if smoker_choice == "Yes" else 0

    input_df = pd.DataFrame([[age,bmi,smoker]],columns=['age','bmi','smoker'])

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Anunnal Premium : ${prediction:.2f}")

    st.info("Note : This is basic ML model estimate based on providing tranning data .")