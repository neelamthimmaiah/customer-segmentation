import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation using K-Means Clustering")

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/Customer Purchasing Behaviors.csv")

data = load_data()

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)

# ── EDA ─────────────────────────────────────────────────────────────────────
st.header("1. Dataset Overview")
st.dataframe(data.head())
st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

st.header("2. Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data, x='age', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data, x='annual_income', ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data, x='purchase_amount', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data, x='loyalty_score', ax=ax)
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 3))
sns.scatterplot(data, x='purchase_frequency', y='loyalty_score', hue='region', ax=ax)
st.pyplot(fig)

# ── Preprocessing ────────────────────────────────────────────────────────────
st.header("3. Preprocessing")

df = data.drop(columns='user_id').copy()
df = pd.get_dummies(df, columns=['region'], dtype='int')

num_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency', 'loyalty_score']
scalers = {}
for col in num_features:
    sc = StandardScaler()
    df[[col]] = sc.fit_transform(df[[col]])
    scalers[col] = sc

st.success("✅ One-hot encoding and standard scaling applied.")
st.dataframe(df.head())

# ── Elbow method ─────────────────────────────────────────────────────────────
st.header("4. Elbow Method")

wcss = []
for i in range(2, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(df)
    wcss.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(2, 11), wcss, marker='*')
ax.set_xlabel("Number of clusters")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")
st.pyplot(fig)

# ── KMeans clustering ─────────────────────────────────────────────────────────
st.header(f"5. K-Means Clustering (k={k})")

model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(df)
labels = model.predict(df)

score = silhouette_score(df, labels)
st.metric("Silhouette Score", f"{score:.4f}")

df1 = data.drop(columns='user_id').copy()
df1['cluster'] = labels

# ── Cluster visualisations ────────────────────────────────────────────────────
st.header("6. Cluster Visualisations")

pairs = [
    ('annual_income', 'purchase_amount'),
    ('annual_income', 'age'),
    ('purchase_frequency', 'loyalty_score'),
]

for x_col, y_col in pairs:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    sns.scatterplot(df1, x=x_col, y=y_col, ax=ax1)
    ax1.set_title("Without clusters")
    sns.scatterplot(df1, x=x_col, y=y_col, hue='cluster', palette='tab10', ax=ax2)
    ax2.set_title(f"With {k} clusters")
    plt.tight_layout()
    st.pyplot(fig)

# ── Cluster summary ───────────────────────────────────────────────────────────
st.header("7. Cluster Summary")
summary = df1.groupby('cluster')[num_features].mean().round(2)
st.dataframe(summary)
