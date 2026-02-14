import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Load the best model (XGBoost) and data
model = joblib.load('model_xgboost.joblib')
data = joblib.load('processed_data.joblib')
X_test = data['X_test']
features = data['features']

# Ensure all data is numeric for SHAP
X_test = X_test.astype(float)

# SHAP Analysis
print("Running SHAP analysis...")
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# 1. SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot (XGBoost)')
plt.savefig('plots/shap_summary_plot.png', bbox_inches='tight')
plt.close()

# 2. SHAP Bar Plot (Global Importance)
plt.figure(figsize=(12, 8))
shap.plots.bar(shap_values, show=False)
plt.title('SHAP Global Feature Importance')
plt.savefig('plots/shap_bar_plot.png', bbox_inches='tight')
plt.close()

# Identify top features
# Extracting mean absolute SHAP values for global importance
vals = np.abs(shap_values.values).mean(0)
feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['feature', 'importance_score'])
feature_importance.sort_values(by=['importance_score'], ascending=False, inplace=True)
feature_importance.to_csv('top_features.csv', index=False)

print("SHAP analysis complete. Plots saved in 'plots' directory.")
print("Top features saved in 'top_features.csv'.")
print("\nTop 10 Features Influencing Churn:")
print(feature_importance.head(10))
