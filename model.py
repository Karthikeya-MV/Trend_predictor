import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.losses import MeanSquaredError


# Load dataset
data = pd.read_csv("Data/twitter_data.csv")

# Encode hashtags
label_encoder = LabelEncoder()
data['hashtag_encoded'] = label_encoder.fit_transform(data['hashtag'])

# Features (hashtag) and target (count)
X = data['hashtag_encoded'].values.reshape(-1, 1)
y = data['count'].values

# Normalize the counts
scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output is a continuous value (count)

# Compile the model
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("models/trend_predictor.h5")