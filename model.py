import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import re
from textblob import TextBlob

# Load dataset
df = pd.read_csv("Data/data.csv")

# Preprocess hashtags
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.lower()

df["cleaned_hashtag"] = df["content"].apply(clean_text)

# Train Word2Vec model
sentences = df["cleaned_hashtag"].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}

# Get sentiment score from text
def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

# Get feature vectors (Word2Vec embedding + sentiment)
def get_vector(text):
    words = text.split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    word2vec_embedding = np.mean(vectors, axis=0) if vectors else np.zeros(100)  # If no known words, use a zero vector
    sentiment_score = get_sentiment(text)  # Get sentiment score
    return np.hstack([word2vec_embedding, sentiment_score])  # Combine Word2Vec embedding and sentiment score

# Apply feature extraction to the dataset
X = np.array(df["cleaned_hashtag"].apply(get_vector).tolist())
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(X)
# Apply K-means clustering (let's assume 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(df['cluster'])
cluster_means = df.groupby('cluster')['score'].mean()
print(cluster_means)
# Map clusters to categories (manual mapping based on cluster means)
def map_cluster_to_category(cluster_id):
    if cluster_id == 0:
        return 'Non-Viral'
    elif cluster_id == 1:
        return 'Medium'
    else:
        return 'Viral'

df['category'] = df['cluster'].apply(map_cluster_to_category)

# Save models
w2v_model.save('models/word2vec.model')
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, 'models/scaler.pkl')
print("Model training completed & saved!")

