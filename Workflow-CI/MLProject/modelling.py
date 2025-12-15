import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

DATA_PATH = "Sales_Transaction_v4a_preprocessed.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["TotalValue_scaled"])
y = df["TotalValue_scaled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    mlflow.log_metric("r2_score", r2)

    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
