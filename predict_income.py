
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib

# Load the trained model, scaler, and encoders
model = tf.keras.models.load_model('dnn_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

# Define the relevant columns and categorical columns
features = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'relationship']
categorical_columns = ['relationship']

# Streamlit application
st.title('Income Prediction App')

# Function to create input fields based on columns
def get_user_input():
    input_data = {}
    for column in features:
        if column in categorical_columns:
            if column in label_encoders:
                options = label_encoders[column].classes_
                input_data[column] = st.selectbox(f"Select {column}", options=options)
            else:
                st.error(f"Label encoder for '{column}' is missing.")
                st.stop()
        else:
            input_data[column] = st.number_input(f"Enter {column}", value=0.0)
    return input_data

# Get user input
user_input = get_user_input()

# Create a Predict button
if st.button('Predict'):
    # Convert the dictionary to a DataFrame
    new_data = pd.DataFrame(user_input, index=[0])

    # Ensure all expected columns are present, fill missing ones with default values
    for column in features:
        if column not in new_data.columns:
            new_data[column] = 0.0  # Default value, adjust if needed

    # Reorder columns to match the expected order
    new_data = new_data[features]

    # Encode the categorical variables using the saved label encoders
    for column in categorical_columns:
        if column in new_data.columns:
            if column in label_encoders:
                new_data[column] = label_encoders[column].transform(new_data[column])
            else:
                st.error(f"Encoder for '{column}' not found.")
                st.stop()

    # Ensure the data types are correct
    for column in features:
        if new_data[column].dtype == 'object':
            st.error(f"Column '{column}' contains non-numeric values after encoding.")
            st.stop()

    # Check if the scaler feature names match
    if not all(col in scaler.feature_names_in_ for col in features):
        st.error("Feature columns do not match the scaler's expected columns.")
        st.stop()

    # Scale the features using the saved scaler
    X_new_data = scaler.transform(new_data)

    # Make predictions
    predictions = model.predict(X_new_data)
    predicted_classes = (predictions > 0.5).astype("int32")

    # Decode the predicted classes back to original labels
    decoded_predictions = le_target.inverse_transform(predicted_classes.ravel())

    # Output the prediction result
    st.write(f"Predicted Target: {decoded_predictions[0]}")

    # Additional information about the prediction
    if decoded_predictions[0] == '0':
        outcome_message = "The income does not exceed the threshold (e.g., income <= $50,000)."
    elif decoded_predictions[0] == '1':
        outcome_message = "The income exceeds the threshold (e.g., income > $50,000)."
    else:
        outcome_message = "Unknown prediction."

    st.write(outcome_message)


