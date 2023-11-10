import streamlit as st
import numpy as np
import pickle

# Load the trained SVM model
model = pickle.load(open('FrancoBraconi_trained_wine_SVMclassification_model.sav', 'rb'))

# Streamlit App
st.title("Wine Quality Prediction App 🍷")
st.subheader("Check the quality of your wine!")

# Image
st.image('wine_fraud.png', use_column_width=True)

# User input for wine characteristics
st.subheader("Wine Characteristics:")
fixed_acidity = st.slider("Fixed Acidity", min_value=0.0, max_value=20.0, value=10.0)
volatile_acidity = st.slider("Volatile Acidity", min_value=0.0, max_value=1.5, value=0.5, step=0.001)
citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
residual_sugar = st.slider("Residual Sugar", min_value=0.0, max_value=100.0, value=10.0, step=0.01)
chlorides = st.slider("Chlorides", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", min_value=0.0, max_value=300.0, value=30.0, step=0.1)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
density = st.slider("Density", min_value=0.0, max_value=1.0, value=0.99, step=0.0001)
pH = st.slider("pH", min_value=0.0, max_value=5.0, value=3.0, step=0.01)
sulphates = st.slider("Sulphates", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
alcohol = st.slider("Alcohol", min_value=5.0, max_value=15.0, value=10.0, step=0.01)
is_white = st.checkbox("Is White?")

# Button to trigger prediction
if st.button("Predict"):
    features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, int(is_white)]).reshape(1, -1)

    # Load the saved scaler and scale the input features
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    features = scaler.transform(features)

    # Predict the wine quality
    prediction = model.predict(features)

    # Display prediction and accuracy level
    st.subheader("Prediction:")
    st.write(f"The predicted wine quality is: {prediction[0]}")

    # Celebrate for Legit result or Alarm for Fraud
    if prediction[0] == "Legit":
        st.success("Congratulations! Your wine is considered Legit! 🎉🍷")
    else:
        st.error("Warning! Your wine is considered Fraud! 🚨🍷")
