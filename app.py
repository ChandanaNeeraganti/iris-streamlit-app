import streamlit as st
import joblib
import numpy as np

# Load the saved KNN model and label encoder
model = joblib.load("knn_iris_model.joblib")
encoder = joblib.load("label_encoder_iris.joblib")

st.title("Iris Flower Species Prediction")

# Input fields for user to enter flower measurements
sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0
)
sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5
)
petal_length = st.number_input(
    "Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4
)
petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2
)

if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict encoded species
    prediction_encoded = model.predict(input_data)

    # Convert prediction back to species name
    prediction_species = encoder.inverse_transform(prediction_encoded)

    st.success(f"Predicted Iris species: {prediction_species[0]}")
