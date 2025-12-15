# automate_Rizki-Muhammad-Syamsi.py

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['CustomerNo'] = df['CustomerNo'].fillna("Unknown").astype(str)
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    df['TotalValue'] = df['Price'] * df['Quantity']

    def remove_outliers_iqr(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[col] >= Q1 - 1.5 * IQR) &
                    (data[col] <= Q3 + 1.5 * IQR)]

    for col in ['Price', 'Quantity', 'TotalValue']:
        df = remove_outliers_iqr(df, col)

    df['CustomerNo_enc'] = LabelEncoder().fit_transform(df['CustomerNo'])
    df['ProductNo_enc'] = LabelEncoder().fit_transform(df['ProductNo'])
    df = pd.get_dummies(df, columns=['Country'])

    scaler = StandardScaler()
    df[['Price_scaled', 'Quantity_scaled', 'TotalValue_scaled']] = scaler.fit_transform(
        df[['Price', 'Quantity', 'TotalValue']]
    )

    df_final = df.select_dtypes(include=['int64', 'float64']).fillna(0)
    return df_final


if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_processed = preprocess_data(input_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_csv, index=False)

    print("✅ Preprocessing selesai, file disimpan di:", output_csv)
