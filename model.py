import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import re

# Load dataset
df = pd.read_csv('Data/data.csv')

# Standardize the target score
scaler = StandardScaler()
df['score'] = scaler.fit_transform(df[['score']])

# Clean text by removing non-ASCII characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

# Clean the text (remove URLs, mentions, hashtags, and punctuation)
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Sentiment Analysis
def get_sentiment(text):
    cleaned_text = clean_text(text)
    sentiment = TextBlob(cleaned_text).sentiment.polarity
    return sentiment

# Apply text cleaning and sentiment extraction
df["content"] = df["content"].apply(remove_non_ascii)
df["sentiment_score"] = df["content"].apply(get_sentiment)


# Features and target
X = df.drop(columns=["content", "score"])
y = df["score"]

# Scale the features (input variables)
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train the model (Neural Network)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save("models/trend_predictor.h5")
import joblib
joblib.dump(scaler, 'models/scaler.pkl')


# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
