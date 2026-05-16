"""
Interactive diabetes risk prediction desktop UI using tkinter (replaces ipywidgets for local use).
Loads diabetes_model.pkl and diabetes_scaler.pkl produced by diabetes_trainer.py. Provides
integer sliders for Pregnancies (0-20), Glucose (0-200), BloodPressure (0-150), and Age (1-100)
with live value display. Validates that Glucose > 0, shows prediction probability, and applies
color-coded output: green for no diabetes detected, red for high risk. Run trainer first.
"""

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "diabetes_scaler.pkl"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "Missing Files",
        "diabetes_model.pkl or diabetes_scaler.pkl not found.\n"
        "Run diabetes_trainer.py first."
    )
    raise SystemExit("Run diabetes_trainer.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

root = tk.Tk()
root.title("Diabetes Risk Predictor")
root.geometry("500x430")
root.resizable(False, False)
root.configure(bg="#f0f4f8")

tk.Label(
    root, text="Diabetes Risk Predictor",
    font=("Helvetica", 16, "bold"), bg="#f0f4f8", fg="#1a3a5c"
).pack(pady=(20, 10))

frame = tk.Frame(root, bg="#f0f4f8")
frame.pack(padx=30, fill="x")

PARAMS = [
    ("Pregnancies",   0,  20,   1),
    ("Glucose",       0, 200, 100),
    ("BloodPressure", 0, 150,  70),
    ("Age",           1, 100,  30),
]
sliders = {}

for label_text, min_v, max_v, default in PARAMS:
    row = tk.Frame(frame, bg="#f0f4f8")
    row.pack(fill="x", pady=5)

    tk.Label(
        row, text=f"{label_text}:", width=16, anchor="w",
        font=("Helvetica", 10), bg="#f0f4f8"
    ).pack(side="left")

    var = tk.IntVar(value=default)
    tk.Scale(
        row, from_=min_v, to=max_v, variable=var, orient="horizontal",
        length=260, bg="#f0f4f8", highlightthickness=0,
        showvalue=True, resolution=1, font=("Helvetica", 9)
    ).pack(side="left")

    sliders[label_text] = var

result_label = tk.Label(
    root, text="Press Predict to see result",
    font=("Helvetica", 13, "bold"), bg="#f0f4f8", fg="#888", pady=12
)
result_label.pack(pady=(15, 5))


def predict():
    glucose = sliders["Glucose"].get()
    if glucose == 0:
        result_label.configure(
            text="Glucose value must be greater than 0", fg="#e67e00"
        )
        return

    features = np.array([[
        sliders["Pregnancies"].get(),
        glucose,
        sliders["BloodPressure"].get(),
        sliders["Age"].get(),
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        result_label.configure(
            text=f"High Risk of Diabetes  ({probability * 100:.1f}% probability)",
            fg="#c0392b"
        )
    else:
        result_label.configure(
            text=f"No Diabetes Detected  ({(1 - probability) * 100:.1f}% confidence)",
            fg="#27ae60"
        )


tk.Button(
    root, text="  Predict  ", command=predict,
    font=("Helvetica", 11, "bold"), bg="#1a3a5c", fg="white",
    relief="flat", padx=16, pady=6, cursor="hand2"
).pack(pady=5)

root.mainloop()
