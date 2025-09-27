import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('../data/smartlight_data.csv')
X = df[['ambient_light', 'motion_detected', 'temperature', 'hour']]
y = df['light_strength']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use
joblib.dump(scaler, '../models/scaler.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Build lightweight model
model = tf.keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# Save model
model.save('../models/model.h5')
print("Model trained and saved to models/model.h5")
