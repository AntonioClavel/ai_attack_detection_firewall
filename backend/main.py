from fastapi import FastAPI
import pandas as pd
import joblib
import json
import xgboost as xgb
import keras
import numpy as np
import os

app = FastAPI()

scaler = joblib.load('models/scaler.pkl')
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)
with open('models/mapping.json', 'r') as f:
    mapping = json.load(f)

nn_model = keras.models.load_model('models/nn_model.keras', compile=False)
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgb_model.json')

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = df[feature_names]
    
    X_scaled = scaler.transform(df)
    
    nn_prob = nn_model.predict(X_scaled, verbose=0)[0][0]
    
    # We use 0.05 as our threshold (reasoning is in the NN training file)
    is_attack = nn_prob > 0.05
    nn_label = "Attack Detected" if is_attack else "Normal/Benign"
    
    xgb_label = "N/A (Normal Traffic)"
    
    # --- XGBOOST ---
    if is_attack:
        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        xgb_preds = xgb_model.predict(dmatrix)
        
        xgb_output_idx = np.argmax(xgb_preds, axis=1)[0]
        
        real_class_id = xgb_output_idx + 1
        
        xgb_label = mapping.get(str(real_class_id), f"Unknown Attack ({real_class_id})")
    return {
        "nn_prediction": nn_label,
        "xgb_prediction": xgb_label
    }