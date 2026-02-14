import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('churn.csv')

# 1. Handle Missing Values
# TotalCharges is often read as an object due to spaces.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill missing TotalCharges with median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 2. Drop unique identifier
df = df.drop('customerID', axis=1)

# 3. Encoding Categorical Data
# Binary encoding for 2-level categories
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encoding for multi-level categories
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Feature Scaling
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Save the preprocessed data and scaler
data = {
    'X_train': X_train_res,
    'X_test': X_test,
    'y_train': y_train_res,
    'y_test': y_test,
    'features': X.columns.tolist()
}
joblib.dump(data, 'processed_data.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Preprocessing complete.")
print(f"Original training set shape: {y_train.shape}")
print(f"Resampled training set shape: {y_train_res.shape}")
