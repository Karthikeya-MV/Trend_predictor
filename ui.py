# ui/app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset and model
data = pd.read_csv("Data/twitter_data.csv")
model = load_model("models/trend_predictor.h5")

# Encode hashtags
label_encoder = LabelEncoder()
data['hashtag_encoded'] = label_encoder.fit_transform(data['hashtag'])

# Normalize counts
scaler = StandardScaler()
scaler.fit(data['count'].values.reshape(-1, 1))

st.title("Hashtag Popularity Predictor")

# User input: Hashtag
user_input = st.text_input("Enter a hashtag (e.g., #AI):")

if st.button("Predict"):
    if user_input:
        # Encode the input hashtag
        try:
            encoded_hashtag = label_encoder.transform([user_input])
            # Predict popularity
            predicted_count = model.predict(np.array([encoded_hashtag]))
            # Inverse transform to get actual count
            actual_count = scaler.inverse_transform(predicted_count.reshape(-1, 1))[0][0]
            st.write(f"Predicted Popularity (Count): {int(actual_count)}")
        except ValueError:
            st.write("Hashtag not found in the dataset.")
    else:
        st.write("Please enter a hashtag.")