# app/main.py
from fastapi import FastAPI
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Load dataset and model
data = pd.read_csv("Data/twitter_data.csv")
model = load_model("models/trend_predictor.h5")

# Encode hashtags
label_encoder = LabelEncoder()
data['hashtag_encoded'] = label_encoder.fit_transform(data['hashtag'])

# Normalize counts
scaler = StandardScaler()
scaler.fit(data['count'].values.reshape(-1, 1))

@app.post("/predict")
def predict(hashtag: str):
    try:
        # Encode the input hashtag
        encoded_hashtag = label_encoder.transform([hashtag])
        # Predict popularity
        predicted_count = model.predict(np.array([encoded_hashtag]))
        # Inverse transform to get actual count
        actual_count = scaler.inverse_transform(predicted_count.reshape(-1, 1))[0][0]
        return {"hashtag": hashtag, "predicted_count": int(actual_count)}
    except ValueError:
        return {"error": "Hashtag not found in the dataset."}