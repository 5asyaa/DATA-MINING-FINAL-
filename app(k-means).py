import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

st.set_page_config(page_title="K-Means Clustering Penjualan", layout="wide")

st.title("K-Means Clustering Penjualan")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        sep=";",
        thousands=".",
        engine="python",
        on_bad_lines="skip"
    )

    st.subheader("Data Awal")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")
    st.text(df.info())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    data = df[['Units_Sold', 'Unit_Price']]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)

    df_clean = df.loc[data.index].reset_index(drop=True)

    st.subheader("Data Setelah Cleaning")
    st.dataframe(data.head())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    st.subheader("Cek NaN Setelah Normalisasi")
    st.write(np.isnan(scaled_data).sum() == 0)

    st.subheader("Pemeriksaan Outlier")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Elbow Method")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1, 11), inertia, marker='o')
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    st.subheader("Evaluasi Silhouette Score")
    sample_data = resample(
        scaled_data,
        n_samples=min(3000, len(scaled_data)),
        random_state=42
    )

    sil_results = {}
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(sample_data)
        score = silhouette_score(sample_data, labels)
        sil_results[k] = score

    st.dataframe(
        pd.DataFrame.from_dict(
            sil_results, orient="index", columns=["Silhouette Score"]
        )
    )

    st.subheader("Clustering Akhir")
    k = st.selectbox("Pilih Jumlah Cluster", options=[2,3,4,5], index=0)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_clean['cluster'] = kmeans.fit_predict(scaled_data)

    st.dataframe(df_clean.head())

    sample_idx = np.random.choice(
        len(scaled_data),
        size=min(3000, len(scaled_data)),
        replace=False
    )

    sil_final = silhouette_score(
        scaled_data[sample_idx],
        df_clean.loc[sample_idx, 'cluster']
    )

    st.subheader("Silhouette Score Akhir")
    st.write(sil_final)

    st.subheader("Visualisasi Cluster")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        data=df_clean,
        x='Units_Sold',
        y='Unit_Price',
        hue='cluster',
        palette='Set2',
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Rata-rata Tiap Cluster")
    summary = df_clean.groupby('cluster')[['Units_Sold', 'Unit_Price']].mean()
    st.dataframe(summary)

    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Hasil Clustering",
        csv,
        "hasil_kmeans_clustering.csv",
        "text/csv"
    )
