import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re

model = load_model("models/trend_predictor.h5")

scaler = joblib.load('models/scaler.pkl')

# Sentiment analysis function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def get_sentiment(text):
    cleaned_text = clean_text(text)
    sentiment = TextBlob(cleaned_text).sentiment.polarity
    return sentiment

# Streamlit Interface
st.title("Hashtag Popularity Predictor")

# User input for hashtag content
user_input = st.text_input("Enter a hashtag (e.g., #AI):")

if st.button("Predict"):
    if user_input:
        # Preprocessing for user input
        sentiment = get_sentiment(user_input)
        features = np.array([[sentiment]])
        
        # Ensure features are in the correct format before scaling

        scaled_features = scaler.transform(features).reshape(-1,1)
    
        # Predict popularity
        predicted_score = model.predict(scaled_features)
    
        # Inverse transform to get actual score
        actual_score = scaler.inverse_transform(predicted_score.reshape(-1, 1))[0][0] 
        # Display the predicted score and category
        st.write(f"Predicted Popularity (Score): {actual_score}")
    else:
        st.write("Please enter a hashtag.")
