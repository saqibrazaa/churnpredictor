from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

# Load model, scaler, and features
model = joblib.load('model_xgboost.joblib')
scaler = joblib.load('scaler.joblib')
data_config = joblib.load('processed_data.joblib')
FEATURES = data_config['features']

# Static files for plots and frontend
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/plots", StaticFiles(directory="plots"), name="plots")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def preprocess_input(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Binary encoding (same as preprocess.py)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = 1 if df[col].iloc[0] == 'Yes' or df[col].iloc[0] == 'Female' else 0
    
    # One-hot encoding categories
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod']
    
    # Create dummies and align with FEATURES
    df_dummies = pd.get_dummies(df, columns=cat_cols)
    
    # Initialize a DF with all features set to 0
    final_df = pd.DataFrame(0, index=[0], columns=FEATURES)
    
    # Fill in the numerical values
    final_df['tenure'] = df['tenure']
    final_df['MonthlyCharges'] = df['MonthlyCharges']
    final_df['TotalCharges'] = df['TotalCharges']
    final_df['SeniorCitizen'] = df['SeniorCitizen']
    final_df['gender'] = df['gender']
    final_df['Partner'] = df['Partner']
    final_df['Dependents'] = df['Dependents']
    final_df['PhoneService'] = df['PhoneService']
    final_df['PaperlessBilling'] = df['PaperlessBilling']
    
    # Fill in dummy values
    for col in df_dummies.columns:
        if col in FEATURES:
            final_df[col] = df_dummies[col]
            
    # Scaling
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    final_df[num_cols] = scaler.transform(final_df[num_cols])
    
    return final_df

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        processed_df = preprocess_input(data)
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]
        
        return {
            "churn": "Yes" if prediction == 1 else "No",
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
