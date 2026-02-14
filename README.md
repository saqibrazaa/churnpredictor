# Telco Customer Churn AI Dashboard

A comprehensive end-to-end customer churn analysis and prediction project featuring a premium interactive web dashboard.

## Features
- **Exploratory Data Analysis**: Visual insights into churn trends, tenure, and service types.
- **Predictive Modeling**: Comparison of Logistic Regression, Random Forest, and XGBoost.
- **Churn Predictor**: Real-time risk analysis for individual customers.
- **Model Explainability**: SHAP plots to visualize global and local feature importance.
- **Business Strategy**: Actionable insights to improve customer retention.

## Tech Stack
- **Backend**: Python, FastAPI, XGBoost, SHAP, Scikit-learn
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript

## Setup and Running Locally
1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic pandas joblib xgboost shap scikit-learn matplotlib seaborn imbalanced-learn
   ```
2. **Run Analysis & Train Models**:
   ```bash
   python eda.py
   python preprocess.py
   python train_models.py
   python interpretability.py
   ```
3. **Launch Dashboard**:
   ```bash
   python app.py
   ```
4. **Access Dashboard**: Open `http://localhost:8000` in your browser.

## Project Structure
- `eda.py`: Generates exploratory visualizations.
- `preprocess.py`: Handles data cleaning, encoding, and SMOTE.
- `train_models.py`: Trains and evaluates prediction models.
- `interpretability.py`: Generates SHAP explainability plots.
- `app.py`: FastAPI backend server.
- `static/`: Frontend dashboard assets.
- `plots/`: Generated analysis visualizations.
- `processed_data.joblib`: Split and preprocessed data.
- `model_xgboost.joblib`: The primary prediction model.

---
*Created with ❤️ by Antigravity*
