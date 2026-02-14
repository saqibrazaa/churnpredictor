import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the dataset
df = pd.read_csv('churn.csv')

# Set style
sns.set(style="whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Churn Distribution')
plt.savefig('plots/churn_distribution.png')
plt.close()

# 2. Tenure Distribution by Churn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, element="step", palette='magma')
plt.title('Tenure Distribution by Churn')
plt.savefig('plots/tenure_churn.png')
plt.close()

# 3. Monthly Charges by Churn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='coolwarm')
plt.title('Monthly Charges Distribution by Churn')
plt.savefig('plots/monthly_charges_churn.png')
plt.close()

# 4. Churn by Contract Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Contract Type')
plt.savefig('plots/contract_churn.png')
plt.close()

# 5. Churn by Internet Service Type
plt.figure(figsize=(10, 6))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set1')
plt.title('Churn by Internet Service Type')
plt.savefig('plots/internet_service_churn.png')
plt.close()

# 6. Service patterns - Stacked bar for multiple services vs Churn
# Identify top service features
services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Analyze TechSupport specifically as it's often a high-churn factor if "No"
plt.figure(figsize=(10, 6))
sns.countplot(x='TechSupport', hue='Churn', data=df, palette='husl')
plt.title('Churn by Tech Support')
plt.savefig('plots/tech_support_churn.png')
plt.close()

print("EDA visualizations generated and saved in 'plots' directory.")
