from fastapi import FastAPI
import pandas as pd
import joblib
import json
import keras
import numpy as np
import os

app = FastAPI()

scaler = joblib.load('models/scaler.pkl')

with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

with open('models/threshold.json', 'r') as f:
    threshold_data = json.load(f)
    best_threshold = threshold_data['binary_threshold']

nn_model = keras.models.load_model('models/nn_model.keras', compile=False)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = df[feature_names]
    
    X_scaled = scaler.transform(df)
    
    nn_prob = nn_model.predict(X_scaled, verbose=0)[0][0]
    
    is_attack = nn_prob > best_threshold
    nn_label = "Attack Detected" if is_attack else "Normal/Benign"
    
    return {
        "prediction": nn_label
    }