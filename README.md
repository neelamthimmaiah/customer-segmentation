# 🛍️ Customer Segmentation using K-Means Clustering

A Streamlit web app that segments customers based on purchasing behaviour using K-Means clustering.

## Features
- Exploratory Data Analysis with interactive charts
- Elbow method to determine optimal number of clusters
- Adjustable `k` via sidebar slider
- Silhouette score evaluation
- Cluster visualisations and summary table

## Dataset
[Customer Purchasing Behaviors](https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors) from Kaggle.

Download the CSV and place it at:
```
data/Customer Purchasing Behaviors.csv
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set main file to `app.py`
4. Click Deploy
