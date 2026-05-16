# ML Concepts Hub

A collection of machine learning and computer vision applications covering recommendation systems, face detection, medical diagnosis, and financial forecasting. Each module is self-contained in its own directory and shares a single virtual environment.

---

## Project Structure

```
Applied_Programming_Lab/
├── venv/                            ← Unified virtual environment
├── requirements.txt
├── recommendation_system/
│   └── recommendation_system.py
├── face_detection/
│   └── face_detection.py
├── diabetes_prediction/
│   ├── diabetes_trainer.py
│   └── diabetes_predictor.py
└── stock_forecasting/
    └── stock_forecasting.py
```

---

## Setup

```bash
# Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

---

## Modules

### 1. Recommendation System
**File:** `recommendation_system/recommendation_system.py`

**Model:** Collaborative Filtering using Cosine Similarity

**How it works:**

User preferences are stored in a user-item matrix where rows are items and columns are users. A rating of `0` means the user has not interacted with that item. The matrix is transposed so each user becomes a row vector, allowing cosine similarity to measure how aligned two users are based on their shared rating patterns. A similarity score of 1 means identical preferences, and 0 means no overlap.

Items are recommended to a target user by finding their most similar users, collecting items those similar users rated highly that the target user has not yet seen, and computing a weighted score based on both the similarity weight and the rating value. The higher the weighted score, the stronger the recommendation.

Rating normalization is applied to correct for user bias — users who habitually rate everything high or low. Each user's average rating is subtracted from their rated items so that scores become relative rather than absolute, making similarity comparisons fairer.

A seaborn heatmap of the full similarity matrix is generated and saved as `similarity_heatmap.png`.

**Run:**
```bash
cd recommendation_system
python recommendation_system.py
```

**Output:** Similarity matrix, top similar users, unrated items, recommendations, normalized matrix, heatmap image.

---

### 2. Face Detection
**File:** `face_detection/face_detection.py`

**Model:** Haar Cascade Classifier (OpenCV pretrained)

**How it works:**

Haar Cascades are machine learning classifiers trained using thousands of positive (face) and negative (non-face) images. OpenCV ships a pretrained classifier file (`haarcascade_frontalface_default.xml`) for frontal face detection.

The image is first converted from BGR (OpenCV default) to grayscale because the cascade operates on intensity patterns, not color. The `detectMultiScale` function scans the image at multiple scales and positions using a sliding window. At each window position, the classifier evaluates whether the region contains a face-like pattern.

- **scaleFactor** controls how much the image is shrunk at each scale step. A value of `1.1` means 10% reduction per step — finer but slower. A larger value like `1.2` is coarser and faster but may miss small faces.
- **minNeighbors** is a threshold that filters weak detections. A high value like `6` gives fewer but more reliable detections. A low value like `3` is more sensitive but may produce false positives.

The script runs three configurations (default, stricter, more sensitive) and saves all annotated results so you can directly compare the effect of parameter tuning.

**Run:**
```bash
cd face_detection
python face_detection.py path/to/your/image.jpg
```

**Output:** Three annotated images saved as PNG files, face count printed for each configuration.

---

### 3. Diabetes Prediction
**Files:** `diabetes_prediction/diabetes_trainer.py` and `diabetes_prediction/diabetes_predictor.py`

**Model:** Logistic Regression with StandardScaler

#### Part A — Trainer (`diabetes_trainer.py`)

**How it works:**

The Pima Indians Diabetes dataset contains 768 patient records with clinical measurements. The dataset is automatically downloaded on first run. Four features are selected for prediction: **Pregnancies**, **Glucose**, **BloodPressure**, and **Age**, with `Outcome` (0 = no diabetes, 1 = diabetes) as the target.

A `StandardScaler` is applied to normalize features so that no single variable dominates due to scale differences (e.g., glucose values up to 200 vs pregnancies up to 17). The data is split 80/20 with stratification to ensure both classes are proportionally represented in both sets. Logistic Regression fits a sigmoid function to estimate the probability of diabetes given the four features. The model and scaler are saved to disk as `.pkl` files using `joblib`.

**Run first:**
```bash
cd diabetes_prediction
python diabetes_trainer.py
```

**Output:** `diabetes.csv` (downloaded), `diabetes_model.pkl`, `diabetes_scaler.pkl`, accuracy score, classification report.

#### Part B — Predictor (`diabetes_predictor.py`)

**How it works:**

Loads the saved model and scaler. Opens a desktop GUI built with `tkinter` providing four integer sliders for each input feature. When Predict is clicked, the slider values are scaled using the saved `StandardScaler` and passed to the loaded `LogisticRegression` model. The model returns both a binary prediction and a probability score. Results are displayed with color-coded output: green for no diabetes detected, red for high risk. Glucose is validated to be greater than zero before prediction.

**Run after trainer:**
```bash
python diabetes_predictor.py
```

**Output:** Interactive desktop window with sliders and real-time prediction.

---

### 4. Stock Price Forecasting
**File:** `stock_forecasting/stock_forecasting.py`

**Model:** Linear Regression (next-day price prediction)

**How it works:**

Historical OHLCV (Open, High, Low, Close, Volume) data for Apple (AAPL), Microsoft (MSFT), and Google (GOOGL) is fetched for the full year 2022 using the `yfinance` library. A new column `Tomorrow` is created by shifting the closing price column by -1, making each row's target the following trading day's close. The last row is dropped because its `Tomorrow` value is NaN (no following day exists in the dataset).

Linear Regression learns the relationship between today's closing price and tomorrow's closing price. This is a simple but effective baseline for short-term price momentum — the model essentially learns that tomorrow's price is likely close to today's. The dataset is split 80/20 chronologically (no shuffling) to prevent data leakage from future into the past.

Performance is measured with:
- **MSE (Mean Squared Error):** measures the average squared difference between predicted and actual prices in USD². Lower is better.
- **R² Score:** measures how much of the variance in next-day prices is explained by today's price. Closer to 1 is better.

All three stocks are compared side by side in a grouped bar chart. Each stock also gets its own closing price chart and prediction chart saved as PNG files.

**Run:**
```bash
cd stock_forecasting
python stock_forecasting.py
```

**Output:** Price charts and prediction plots per stock, cross-stock comparison bar chart (`stocks_comparison.png`), MSE and R² scores printed per stock.

---

## Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, DataFrames |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | Cosine similarity, Logistic/Linear Regression, metrics |
| `seaborn` | Heatmap visualization |
| `matplotlib` | All plotting and chart output |
| `opencv-python` | Image loading, grayscale conversion, Haar cascade detection |
| `joblib` | Saving and loading trained model and scaler |
| `yfinance` | Downloading historical stock market data |
| `requests` | Auto-downloading the diabetes dataset |

---

## Notes

- Run `diabetes_trainer.py` before `diabetes_predictor.py` — the predictor requires the saved model files.
- `face_detection.py` requires an image path as a command-line argument. Use any JPG or PNG containing faces.
- All plots are displayed interactively and saved as PNG files in the working directory.
- The diabetes dataset and stock data are fetched automatically on first run — an internet connection is required.
