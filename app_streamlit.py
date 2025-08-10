# app_streamlit.py
# Streamlit App: Customer Segmentation + Market Basket + BI Summary (ROI)
# ----------------------------------------------------------------------
# Usage:
#   streamlit run app_streamlit.py
#
# Notes:
# - Loads "synthetic_retail_transactions.csv" automatically (or set env RETAIL_CSV_PATH).
# - Implements:
#   • Customer Segmentation Dashboard (2D/3D, stats, lookup)
#   • Market Basket Analysis (rules explorer, recommender, visuals)
#   • BI Summary with ROI projections

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
)
from mlxtend.frequent_patterns import apriori, association_rules

# Optional (network graph)
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Analytics: Segmentation & Market Basket",
                   layout="wide")

# ------------------------------- Helpers --------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    if "Amount" not in df.columns:
        df["Amount"] = df["Quantity"] * df["UnitPrice"]
    # basic cleaning
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    return df

@st.cache_data(show_spinner=False)
def build_rfm_extras(df: pd.DataFrame) -> pd.DataFrame:
    current_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (current_date - x.max()).days,  # Recency
        "InvoiceNo": "nunique",                                   # Frequency
        "Amount": "sum"                                           # Monetary
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # extras
    rfm = (
        rfm.merge(df.groupby("CustomerID")["Amount"].mean().reset_index(name="AvgOrderValue"), on="CustomerID")
           .merge(df.groupby("CustomerID")["ProductID"].nunique().reset_index(name="ProductDiversity"), on="CustomerID")
           .merge(df.groupby("CustomerID")["Quantity"].sum().reset_index(name="TotalQuantity"), on="CustomerID")
           .merge(df.groupby("CustomerID")["Quantity"].mean().reset_index(name="AvgQuantityPerTransaction"), on="CustomerID")
           .merge(df.groupby("CustomerID")["Category"].nunique().reset_index(name="CategoryDiversity"), on="CustomerID")
           .merge(df.groupby("CustomerID")["InvoiceDate"].agg(lambda x: (x.max() - x.min()).days).reset_index(name="PurchaseSpan"), on="CustomerID")
           .merge(df.groupby("CustomerID")["InvoiceDate"].apply(lambda x: (x.sort_values().diff().mean().days) if len(x) > 1 else np.nan)
                      .reset_index(name="AvgDaysBetweenPurchases"), on="CustomerID")
    )
    return rfm

CLUSTER_FEATURES = [
    "Recency","Frequency","Monetary","AvgOrderValue","ProductDiversity",
    "TotalQuantity","AvgQuantityPerTransaction","CategoryDiversity",
    "PurchaseSpan","AvgDaysBetweenPurchases"
]

@st.cache_data(show_spinner=False)
def scale_and_filter(rfm: pd.DataFrame):
    X = rfm[CLUSTER_FEATURES].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median(numeric_only=True))

    # IQR outlier filter
    Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask = ~((X < lower) | (X > upper)).any(axis=1)
    X_clean = X.loc[mask].copy()
    rfm_clean = rfm.loc[mask].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return X_clean, rfm_clean, scaler, X_scaled

def fit_kmeans_alt(X_scaled, rfm_clean, k=5, seed=42):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed)
    clusters = kmeans.fit_predict(X_scaled)
    rfm_k = rfm_clean.copy()
    rfm_k["Cluster"] = clusters

    # Alt methods
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    agg_labels = agg.fit_predict(X_scaled)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    gmm_labels = gmm.fit_predict(X_scaled)

    # metrics
    s_km = silhouette_score(X_scaled, clusters)
    s_agg = silhouette_score(X_scaled, agg_labels)
    s_gmm = silhouette_score(X_scaled, gmm_labels)
    ch_km = calinski_harabasz_score(X_scaled, clusters)
    db_km = davies_bouldin_score(X_scaled, clusters)
    ari_km_agg = adjusted_rand_score(clusters, agg_labels)
    ari_km_gmm = adjusted_rand_score(clusters, gmm_labels)
    ari_agg_gmm = adjusted_rand_score(agg_labels, gmm_labels)

    metrics = {
        "silhouette": {"KMeans": s_km, "Agglomerative": s_agg, "GMM": s_gmm},
        "calinski_harabasz": ch_km,
        "davies_bouldin": db_km,
        "ari": {"KM_vs_AGG": ari_km_agg, "KM_vs_GMM": ari_km_gmm, "AGG_vs_GMM": ari_agg_gmm},
    }
    return kmeans, clusters, agg_labels, gmm_labels, rfm_k, metrics

# Market Basket pieces
def build_boolean_basket(df_in: pd.DataFrame, product_col: str):
    sub = df_in.loc[:, ["InvoiceNo", product_col, "Quantity"]].copy()
    sub = sub.dropna(subset=["InvoiceNo", product_col, "Quantity"])
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce")
    sub = sub.dropna(subset=["Quantity"])
    sub = sub[sub["Quantity"] > 0]
    basket = sub.groupby(["InvoiceNo", product_col])["Quantity"].sum().unstack(fill_value=0)
    return basket.gt(0), product_col

def mine_rules(basket_bool: pd.DataFrame, min_support=0.02, min_conf=0.30):
    itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    # keep support filter
    rules = rules.query("support >= @min_support").copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: list(s))
    rules["consequents"] = rules["consequents"].apply(lambda s: list(s))
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    return rules, itemsets

def filter_rules(rules_df, include=None, exclude=None,
                 min_support=0.02, min_conf=0.30, min_lift=1.10,
                 max_antecedent_len=None):
    r = rules_df.copy()
    r = r[(r["support"] >= min_support) & (r["confidence"] >= min_conf) & (r["lift"] >= min_lift)]
    if max_antecedent_len is not None:
        r = r[r["antecedent_len"] <= int(max_antecedent_len)]
    if include:
        inc = set(include); r = r[r["antecedents"].apply(lambda ants: inc.issubset(set(ants)))]
    if exclude:
        exc = set(exclude)
        r = r[~r["antecedents"].apply(lambda ants: bool(set(ants) & exc))]
        r = r[~r["consequents"].apply(lambda cons: bool(set(cons) & exc))]
    return r.sort_values(["lift","confidence","support"], ascending=False)

def recommend_from_rules(products, rules_df, top_n=10):
    prods = set(products)
    if not prods:
        return pd.DataFrame(columns=["consequent","support","confidence","lift"])
    mask = rules_df["antecedents"].apply(lambda ants: set(ants).issubset(prods))
    recs = (rules_df[mask][["consequents","support","confidence","lift"]]
            .explode("consequents")
            .rename(columns={"consequents":"consequent"}))
    recs = recs[~recs["consequent"].isin(prods)]
    return recs.sort_values(["lift","confidence","support"], ascending=False).head(top_n).reset_index(drop=True)

def draw_rules_network_matplotlib(rules_df, top_n=20):
    if not HAS_NX or rules_df.empty:
        return None
    r = rules_df.sort_values(["lift","confidence"], ascending=False).head(top_n)
    G = nx.DiGraph()
    def lbl(items): return " + ".join(map(str, items))
    for _, row in r.iterrows():
        a, c = lbl(row["antecedents"]), lbl(row["consequents"])
        G.add_node(a, kind="a"); G.add_node(c, kind="c")
        G.add_edge(a, c, lift=float(row["lift"]))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=900, font_size=8, alpha=0.85,
            node_color=["#4c72b0" if G.nodes[n]["kind"]=="a" else "#dd8452" for n in G.nodes()])
    st.pyplot(plt.gcf())
    plt.close()

def build_roi_table(rules_df, df, product_col, eligible=10_000, margin=0.35, cost=1500.0, top_n=10):
    if rules_df.empty:
        return pd.DataFrame()
    basket_bool, _ = build_boolean_basket(df, product_col)
    prevalence = basket_bool.mean(axis=0).to_dict()
    rows = []
    r = rules_df.sort_values(["lift","confidence"], ascending=False).head(top_n)
    for _, row in r.iterrows():
        cons = list(row["consequents"])
        base_attach = float(np.mean([prevalence.get(c, 0.0) for c in cons]))
        cons_prices = df[df[product_col].isin(cons)]["UnitPrice"]
        avg_price = float(cons_prices.mean()) if not cons_prices.empty else float(df["UnitPrice"].mean())
        lift = float(row["lift"])
        inc_attach = max(lift - 1, 0) * base_attach
        incr_rev  = eligible * inc_attach * avg_price * margin
        roi       = (incr_rev - cost) / cost if cost else np.nan
        rows.append({
            "antecedents": ", ".join(row["antecedents"]),
            "consequents": ", ".join(cons),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": lift,
            "base_attach_%": base_attach * 100,
            "incremental_attach_%": inc_attach * 100,
            "avg_consequent_price": avg_price,
            "eligible_baskets": int(eligible),
            "gross_margin": float(margin),
            "campaign_cost": float(cost),
            "expected_incremental_revenue": incr_rev,
            "ROI": roi
        })
    return pd.DataFrame(rows)

# ------------------------------ Sidebar --------------------------------------
st.sidebar.title("CIS 9660 • Data Mining • Project #2")

# Load data silently (no upload/path UI)
DATA_PATH = os.getenv("RETAIL_CSV_PATH", "synthetic_retail_transactions.csv")
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at '{DATA_PATH}'. Place the file next to this app or set RETAIL_CSV_PATH.")
    st.stop()
df = load_data(DATA_PATH)

# Product column dropdown (auto-detected)
candidate_product_cols = [c for c in ["ProductName", "Description", "ProductID"] if c in df.columns]
if not candidate_product_cols:
    st.error("No product column found. Add one of: ProductName, Description, ProductID")
    st.stop()
PRODUCT_COL = st.sidebar.selectbox("Product column", options=candidate_product_cols, index=0)

# Clustering + rules controls
k_value = st.sidebar.slider("K for clustering", 2, 10, 5, 1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Rules thresholds**")
min_support = st.sidebar.slider("Min support", 0.01, 0.10, 0.02, 0.01)
min_conf    = st.sidebar.slider("Min confidence", 0.10, 0.90, 0.30, 0.05)
min_lift    = st.sidebar.slider("Min lift", 1.0, 3.0, 1.10, 0.05)

st.sidebar.markdown("---")
eligible = st.sidebar.number_input("Eligible baskets (ROI)", value=10_000, step=1000)
margin   = st.sidebar.number_input("Gross margin (0-1)", value=0.35, step=0.05, min_value=0.0, max_value=1.0)
cost     = st.sidebar.number_input("Campaign cost ($)", value=1500.0, step=100.0)

# --------------------------- Compute RFM & Clusters ---------------------------
rfm = build_rfm_extras(df)
X_clean, rfm_clean, scaler, X_scaled = scale_and_filter(rfm)
kmeans, clusters, agg_labels, gmm_labels, rfm_k, metrics = fit_kmeans_alt(X_scaled, rfm_clean, k=k_value, seed=seed)

# PCA once for visualizations
pca = PCA(n_components=3).fit(X_scaled)
X_pca = pca.transform(X_scaled)
exp_var = pca.explained_variance_ratio_

# ------------------------------- Tabs ----------------------------------------
tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Market Basket Analysis", "BI Summary & ROI"])

# ================================ TAB 1 ======================================
with tab1:
    st.subheader("Customer Segmentation Dashboard")

    colA, colB = st.columns([3,2])
    with colA:
        st.markdown("**2D PCA Scatter**")
        df_plot = pd.DataFrame({
            "PC1": X_pca[:,0], "PC2": X_pca[:,1],
            "Cluster": clusters.astype(int), "CustomerID": rfm_clean["CustomerID"].values
        })
        fig2d = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
