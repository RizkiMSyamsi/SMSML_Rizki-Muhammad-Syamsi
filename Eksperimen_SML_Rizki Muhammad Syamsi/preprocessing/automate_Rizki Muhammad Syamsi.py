import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fungsi untuk melakukan seluruh tahap preprocessing
def preprocess_data(df):
    df_clean = df.copy()  # copy dataframe
    df_clean.drop_duplicates(inplace=True)  # drop duplikat
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')  # convert tanggal
    df_clean = df_clean.dropna(subset=['Date'])  # drop tanggal invalid
    df_clean['CustomerNo'] = df_clean['CustomerNo'].fillna("Unknown").astype(str)  # isi missing customer
    df_clean = df_clean[df_clean['Quantity'] > 0]  # hapus pembatalan
    df_clean = df_clean[df_clean['Price'] > 0]  # hapus harga salah
    df_clean['TotalValue'] = df_clean['Price'] * df_clean['Quantity']  # hitung total

    def remove_outliers_iqr(df, col):  # hapus outlier dengan IQR
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    for col in ['Price', 'Quantity', 'TotalValue']:  # terapkan outlier removal
        df_clean = remove_outliers_iqr(df_clean, col)

    le_customer = LabelEncoder()  # encode customer
    df_clean['CustomerNo_enc'] = le_customer.fit_transform(df_clean['CustomerNo'])

    le_product = LabelEncoder()  # encode product
    df_clean['ProductNo_enc'] = le_product.fit_transform(df_clean['ProductNo'])

    df_clean = pd.get_dummies(df_clean, columns=['Country'], prefix='Country')  # one-hot encode country

    scaler = StandardScaler()  # scaling numerik
    df_clean[['Price_scaled', 'Quantity_scaled', 'TotalValue_scaled']] = scaler.fit_transform(
        df_clean[['Price', 'Quantity', 'TotalValue']]
    )

    df_clean['Price_bin'] = pd.qcut(df_clean['Price'], q=4, labels=['Very Low', 'Low', 'Medium', 'High'])  # bin price
    df_clean['TotalValue_bin'] = pd.qcut(df_clean['TotalValue'], q=4, labels=['Very Low', 'Low', 'Medium', 'High'])  # bin total

    le_bin = LabelEncoder()  # encode binning
    df_clean['Price_bin_enc'] = le_bin.fit_transform(df_clean['Price_bin'])
    df_clean['TotalValue_bin_enc'] = le_bin.fit_transform(df_clean['TotalValue_bin'])

    return df_clean  # kembalikan dataset akhir


# ==== BAGIAN PROSES OTOMATIS ====

# Load file raw dataset
df_raw = pd.read_csv("../Sales Transaction v.4a_raw.csv")  # sesuaikan path

# Jalankan preprocessing
df_processed = preprocess_data(df_raw)

# Simpan output
df_processed.to_csv("Sales_Transaction_v4a_preprocessed.csv", index=False)
print("Preprocessing selesai — file disimpan sebagai Sales_Transaction_v4a_preprocessed.csv")
