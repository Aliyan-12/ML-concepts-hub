"""
Logistic Regression trainer for diabetes risk prediction on the Pima Indians Diabetes dataset.
Auto-downloads diabetes.csv if absent, selects 4 features (Pregnancies, Glucose, BloodPressure,
Age), applies StandardScaler, performs 80/20 stratified train/test split, trains the model,
evaluates with accuracy and a full classification report, then saves both the fitted model and
scaler as diabetes_model.pkl and diabetes_scaler.pkl for use by diabetes_predictor.py.
"""

import os
from pathlib import Path
import requests
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "diabetes.csv"
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "diabetes_scaler.pkl"
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

if not DATASET_PATH.exists():
    print("Downloading diabetes.csv ...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    DATASET_PATH.write_bytes(response.content)
    print(f"Saved: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print("=== Dataset Preview ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Class distribution:\n{df['Outcome'].value_counts()}\n")

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'Age']
X = df[FEATURES]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
print("=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved : {MODEL_PATH}")
print(f"Scaler saved: {SCALER_PATH}")
print("\nRun diabetes_predictor.py to launch the interactive prediction UI.")
