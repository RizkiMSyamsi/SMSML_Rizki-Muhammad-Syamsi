import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score



# CONFIG
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Sales Prediction - Ridge Tuning")



# LOAD DATA
df = pd.read_csv(
    r"D:\Kuliah\Semester 7\Asah\Tugas Project\SMSML_Rizki-Muhammad-Syamsi\Membangun_model\Sales Transaction v.4a_preprocessing.csv",
    low_memory=False
).infer_objects()



# FEATURE & TARGET
y = df['TotalValue_scaled']

X = df.drop(columns=[
    'TransactionNo',
    'Date',
    'ProductName',
    'Price_bin',
    'TotalValue_bin',
    'TotalValue',
    'TotalValue_scaled'
], errors='ignore')

X = X.select_dtypes(include=['int64', 'float64'])

# convert to float (hindari warning mlflow)
X = X.astype("float64")

# fill missing value
X = X.fillna(0)



# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# GRID SEARCH
param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)



# TRAINING + LOGGING
with mlflow.start_run(run_name="Ridge Regression - GridSearch"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameter terbaik
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    # Log metric
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log model + signature
    mlflow.sklearn.log_model(
        best_model,
        "model",
        input_example=X_train.iloc[:5]
    )

    print("\n✅ RIDGE TUNING SUCCESS")
    print("Best Parameter:", best_params)
    print("MSE :", mse)
    print("R2  :", r2)
