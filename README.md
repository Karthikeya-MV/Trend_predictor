# Social Media Trend Predictor
```Author: Mantripragada Venkata Karthikeya```
---
## Overview
This project predicts trending hashtags by scraping data from Twitter and Reddit, preprocessing the text, and training predictive models (**Word2Vec, Sentiment Polarity and K-Means Clustering**). The solution is deployed features a user-friendly interface via **Streamlit**.

---

## Features
- **Automated Data Collection**: Scrapes trending hashtags from Twitter & Reddit.
- **Advanced NLP Processing**: Cleans and preprocesses text data and Word Embedding.
- **Predictive Modeling**: Uses **Word2Vec Gensim embeddings and Sentiment Polarity** (deep learning) and **K-Means Clusturing** (For Popularity classification).
- **Interactive User Interface**: Choose between **Streamlit**.
---

## Project Structure
```
trend_predictor/
│── Data/
    │── data.csv
    │── other files used in preprocessing
│── models/
    │── saved models after training
│── preprocess.py     # Text preprocessing
│── model.py          # Training
│── ui.py             # Streamlit UI
│── requirements.txt  # Dependencies
│── README.md         # Documentation
```

---

## Installation
### **1️ Install Dependencies**
Ensure Python (3.8+) is installed, then run:
```bash
pip install -r requirements.txt
```

---

## Running the Project
### **2 Preprocess Data**
Clean and prepare text data:
```bash
python preprocess.py
```
**Output:** `Data/data.csv`

### **3 Train Models**
Train model for prediction:
```bash
python model.py
```
**Output:** `models/word2vec.model and models/kmeans.pkl`

### **4 Run UI**
#### **Streamlit UI**
```bash
streamlit run ui.py
```
**Visit:** `http://localhost:8501`

---

## Conclusion
This project provides a complete pipeline for predicting social media trends using **data scraping, NLP, deep learning**. It is designed to be scalable, efficient, and easy to use.
