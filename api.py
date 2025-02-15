from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re

app = FastAPI()

# Load the model and necessary components
model = load_model("models/trend_predictor.h5")
scaler = StandardScaler()
vectorizer = TfidfVectorizer(max_features=10, stop_words='english')

# Preprocess the scaler and vectorizer
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

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

# Load and prepare the dataset for vectorizer
import pandas as pd
df = pd.read_csv("Data/data.csv")
vectorizer.fit(df["content"])

# Define request body structure
class PredictRequest(BaseModel):
    content: str

@app.post("/predict")
def predict_score(request: PredictRequest):
    content = request.content
    sentiment = get_sentiment(content)
    content_tfidf = vectorizer.transform([content]).toarray()

    # Combine sentiment with TF-IDF features
    features = np.array([[sentiment] + list(content_tfidf[0])])

    # Scale the features
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    # Predict the score
    predicted_score = model.predict(features_scaled)[0]
    
    # Inverse transform to get the original score
    actual_score = scaler.inverse_transform(predicted_score.reshape(-1, 1))[0][0]

    return {"predicted_score": actual_score}
