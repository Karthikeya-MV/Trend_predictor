import streamlit as st
import numpy as np
import joblib
from gensim.models import Word2Vec
from model import clean_text, get_sentiment

# Load the trained models
word2vec_model = Word2Vec.load("models/word2vec.model")
model = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")


# Function to get Word2Vec embedding
def get_word2vec_embedding(text):
    words = clean_text(text).split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average vector representation
    else:
        return np.zeros(word2vec_model.vector_size)  # Zero vector if no known words

# Streamlit UI
st.title("📊 Hashtag Popularity Predictor")

# User input
user_input = st.text_input("Enter a hashtag (e.g., #AI):")

if st.button("Predict"):
    if user_input:
        # Extract features
        sentiment_score = get_sentiment(user_input)
        word2vec_embedding = get_word2vec_embedding(user_input)

        # Combine features
        features = np.hstack([sentiment_score, word2vec_embedding]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        # Predict popularity
        cluster_id = model.predict(scaled_features)
        # Categorize results
        if cluster_id[0]==2:
            category = "🔥 Viral 🚀"
        elif cluster_id[0]==1:
            category = "📈 Medium Reach"
        else:
            category = "📉 Low Reach"
        # Display results
        st.write(f"**Category:** {category}")
    else:
        st.write("⚠️ Please enter a hashtag.")
