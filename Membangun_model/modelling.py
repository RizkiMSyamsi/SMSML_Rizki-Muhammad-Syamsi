import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# LOCAL MLFLOW CONFIG (SQLite)
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File database MLflow (akan dibuat otomatis)
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

# Folder untuk artifacts
ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns")

# Set tracking URI ke SQLite lokal
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

# Set experiment (akan tersimpan di mlflow.db)
mlflow.set_experiment("Sales - Linear Regression (Local)")

print(" MLflow Tracking URI:", mlflow.get_tracking_uri())
print(" Artifact location :", ARTIFACT_ROOT)

# ============================
# LOAD DATA (RELATIVE PATH)
# ============================

DATA_PATH = os.path.join(BASE_DIR, "Sales Transaction v.4a_preprocessing.csv")
df = pd.read_csv(DATA_PATH, low_memory=False).infer_objects()

# ============================
# CLEAN FEATURE & TARGET
# ============================

# Target
y = df["TotalValue_scaled"]

# Drop kolom yang tidak digunakan
X = df.drop(
    columns=[
        "TransactionNo",
        "Date",
        "ProductName",
        "Price_bin",
        "TotalValue_bin",
        "TotalValue",
        "TotalValue_scaled",
    ],
    errors="ignore",
)

# Hanya ambil kolom numerik
X = X.select_dtypes(include=["int64", "float64"])

# Konversi ke float
X = X.astype("float64")

# Hilangkan NaN
X = X.fillna(0)

print(" Feature shape:", X.shape)

# ============================
# SPLIT DATA
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ============================
# MLFLOW AUTOLOG
# ============================

mlflow.sklearn.autolog()

# ============================
# TRAINING
# ============================

with mlflow.start_run(run_name="Linear Regression - Sales Prediction (Local)"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n TRAINING SUCCESS (LOCAL)")
    print("MSE :", mse)
    print("R2  :", r2)
    print(" Logged to LOCAL MLflow (mlflow.db)")
