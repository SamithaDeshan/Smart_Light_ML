import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

# Load model and scaler
model = tf.keras.models.load_model('../models/model.h5')
scaler = joblib.load('../models/scaler.pkl')

# Load dataset
df = pd.read_csv('../data/smartlight_data.csv')
X = df[['ambient_light', 'motion_detected', 'temperature', 'hour']]
y = df['light_strength']

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Plot
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Light Strength")
plt.ylabel("Predicted Light Strength")
plt.title("Prediction Accuracy")
plt.show()
