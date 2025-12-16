import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("Analisis Data Motor Bekas Konsumsi BBM")

df = pd.read_csv("motor_second_dataset_r2_07_08.csv")

st.subheader("Dataset")
st.dataframe(df.head())
st.write("Jumlah baris:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])

df = df.dropna().drop_duplicates()

encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("konsumsiBBM", axis=1)
y = df["konsumsiBBM"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_reg.fit(X_train, y_train)

y_pred = gb_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluasi Model")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("R²", round(r2, 3))

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred)
ax1.set_xlabel("Nilai Aktual")
ax1.set_ylabel("Nilai Prediksi")
ax1.set_title("Aktual vs Prediksi Konsumsi BBM")
st.pyplot(fig1)

hasil = X_test.copy()
hasil["BBM_Aktual"] = y_test.values
hasil["BBM_Prediksi"] = y_pred

q1 = hasil["BBM_Prediksi"].quantile(0.33)
q2 = hasil["BBM_Prediksi"].quantile(0.66)

csv = hasil.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Hasil Prediksi",
    data=csv,
    file_name="hasil_prediksi_konsumsi_BBM.csv",
    mime="text/csv"
)
