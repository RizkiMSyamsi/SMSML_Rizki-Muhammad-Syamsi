import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# CONFIG
# Gunakan SQLite sebagai backend MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Buat / set experiment
mlflow.set_experiment("Sales_Linear_Regression")


# LOAD DATA

df = pd.read_csv(
    r"D:\Kuliah\Semester 7\Asah\Tugas Project\SMSML_Rizki-Muhammad-Syamsi\Membangun_model\Sales Transaction v.4a_preprocessing.csv",
    low_memory=False
).infer_objects()



# CLEAN FEATURE & TARGET

# Target
y = df['TotalValue_scaled']

# Drop kolom yang tidak dipakai
X = df.drop(columns=[
    'TransactionNo',
    'Date',
    'ProductName',
    'Price_bin',
    'TotalValue_bin',
    'TotalValue',
    'TotalValue_scaled'
], errors='ignore')

# Ambil kolom numerik saja
X = X.select_dtypes(include=['int64', 'float64'])

# Ubah semuanya ke float (hindari warning MLflow)
X = X.astype("float64")

# Hilangkan NaN
X = X.fillna(0)



# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# AUTLOG MLFLOW

mlflow.sklearn.autolog()



# TRAINING

with mlflow.start_run(run_name="Linear Regression - Sales Prediction"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log model + signature
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train.iloc[:5]
    )

    print("\n✅ TRAINING SUCCESS")
    print("MSE :", mse)
    print("R2  :", r2)
