# Social Media Trend Predictor
```Author: Mantripragada Venkata Karthikeya```
---
## Overview
This project predicts trending hashtags by scraping data from Twitter and Reddit, preprocessing the text, and training predictive models (**LSTM & Prophet**). The solution is deployed using **FastAPI** and features a user-friendly interface via **Streamlit**.

---

## Features
- **Automated Data Collection**: Scrapes trending hashtags from Twitter & Reddit.
- **Advanced NLP Processing**: Cleans and preprocesses text data.
- **Predictive Modeling**: Uses **LSTM** (deep learning) and **Prophet** (time-series forecasting).
- **Scalable API Deployment**: Powered by **FastAPI**.
- **Interactive User Interface**: Choose between **Streamlit**.
---

## Project Structure
```
trend_predictor/
│── api.py            # FastAPI backend
│── preprocess.py     # Text preprocessing
│── model.py          # LSTM & Prophet models
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
Train LSTM & Prophet for prediction:
```bash
python model.py
```
**Output:** `models/trend_predictor.h5`

### **4 Start FastAPI Backend**
Launch the API server:
```bash
uvicorn api.main:app --reload
```
**API Running at:** `http://127.0.0.1:8000`
**Test:** Open `http://127.0.0.1:8000/predict?hashtag=#Example`

### **5 Run UI**
#### **Streamlit UI**
```bash
streamlit run ui.py
```
**Visit:** `http://localhost:8501`

---

## Conclusion
This project provides a complete pipeline for predicting social media trends using **data scraping, NLP, deep learning, and time-series forecasting**. It is designed to be scalable, efficient, and easy to use.

