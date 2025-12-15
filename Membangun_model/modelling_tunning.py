import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)

from dotenv import load_dotenv
load_dotenv()


# =========================================================
# USER CONFIG
# =========================================================

EXPERIMENT_NAME = "Sales Prediction - Ridge Tuning (final)"

DROP_COLUMNS = [
    "TransactionNo",
    "Date",
    "ProductName",
    "Price_bin",
    "TotalValue_bin",
    "TotalValue",
]

TARGET_COL = "TotalValue_scaled"
CSV_FILENAME = "Sales Transaction v.4a_preprocessing.csv"


# =========================================================
# PATH SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, CSV_FILENAME)

MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# =========================================================
# HELPER
# =========================================================

def warn_and_exit(msg):
    print("ERROR:", msg)
    sys.exit(1)

def print_header(title):
    print("\n" + "="*6 + " " + title + " " + "="*6)


# =========================================================
# 1. AUTO MLFLOW MODE (LOCAL / DAGSHUB)
# =========================================================

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

MODE = "dagshub" if DAGSHUB_USERNAME and DAGSHUB_TOKEN else "local"

print_header("MLFLOW SETUP")

if MODE == "local":
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    print("MLflow MODE  : LOCAL")
    print("Tracking URI:", mlflow.get_tracking_uri())

else:
    DAGSHUB_REPO = "SMSML_Rizki-Muhammad-Syamsi"

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )

    print("MLflow MODE  : DAGSHUB")
    print("Username    :", DAGSHUB_USERNAME)
    print("Tracking URI:", mlflow.get_tracking_uri())


# =========================================================
# 2. SET EXPERIMENT
# =========================================================

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)


# =========================================================
# 3. LOAD DATA
# =========================================================

if not os.path.exists(DATA_PATH):
    warn_and_exit(f"File tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False).infer_objects()

print_header("DATA")
print("Shape:", df.shape)


# =========================================================
# 4. PREPROCESSING
# =========================================================

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
df = df.fillna(0)

if TARGET_COL not in df.columns:
    warn_and_exit("Target tidak ditemukan")

df_num = df.select_dtypes(include=[np.number])


# =========================================================
# 5. SPLIT DATA
# =========================================================

y = df_num[TARGET_COL]
X = df_num.drop(columns=[TARGET_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================================
# 6. GRID SEARCH RIDGE
# =========================================================

param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)


# =========================================================
# 7. TRAINING + MANUAL LOGGING
# =========================================================

with mlflow.start_run(run_name="Ridge Regression - GridSearch"):

    print_header("TRAINING")

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    n, p = X_test.shape
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    # ===== LOG PARAM & METRIC =====
    mlflow.log_params(best_params)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("explained_variance", explained_var)
    mlflow.log_metric("adjusted_r2", adjusted_r2)
    mlflow.log_metric("max_error", max_err)

    # ===== ARTIFACT: MODEL =====
    model_path = os.path.join(ARTIFACTS_DIR, "ridge_model.pkl")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    # ===== ARTIFACT: PREDICTION =====
    pred_path = os.path.join(ARTIFACTS_DIR, "prediction.csv")
    pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    }).to_csv(pred_path, index=False)
    mlflow.log_artifact(pred_path)

    # ===== ARTIFACT: PLOT =====
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()

    plot_path = os.path.join(ARTIFACTS_DIR, "actual_vs_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)

    # ===== ARTIFACT: REPORT =====
    report_path = os.path.join(ARTIFACTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write("Ridge Regression GridSearch Report\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"MAPE: {mape}\n")
        f.write(f"Explained Variance: {explained_var}\n")
        f.write(f"Adjusted R2: {adjusted_r2}\n")
        f.write(f"Max Error: {max_err}\n")

    mlflow.log_artifact(report_path)

    print("\nRUN COMPLETE")
    print("Tracking URI:", mlflow.get_tracking_uri())
