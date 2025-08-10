# app_streamlit.py
# Streamlit App: Customer Segmentation + Market Basket + BI Summary (ROI)
# Usage:
#   streamlit run app_streamlit.py
#
# Notes:
# - Reads "synthetic_retail_transactions.csv" from the repo root.
# - Auto-detects the product column: ProductName, then Description, then ProductID.
# - Tabs:
#   1) Customer Segmentation (2D/3D, stats, lookup)
#   2) Market Basket (simple recommender + optional filters + visuals)
#   3) BI Summary and ROI

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
)
from mlxtend.frequent_patterns import apriori, association_rules

# Optional network graph
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Analytics: Segmentation and Market Basket", layout="wide")


# ------------------------------- Helpers --------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    # basics
    if "Amount" not in df.columns:
        df["Amount"] = df["Quantity"] * df["UnitPrice"]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    return df


def ensure_product_col(df: pd.DataFrame) -> str:
    """Auto-pick the first matching product column without exposing a UI."""
    for c in ["ProductName", "Description", "ProductID"]:
        if c in df.columns:
            return c
    raise ValueError("No product column found. Add one of: ProductName, Description, ProductID")


@st.cache_data(show_spinner=False)
def build_rfm_extras(df: pd.DataFrame) -> pd.DataFrame:
    current_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (current_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Amount": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

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

    # IQR filter
    Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)

    X_clean = X.loc[mask].copy()
    rfm_clean = rfm.loc[mask].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return X_clean, rfm_clean, scaler, X_scaled


def fit_kmeans_alt(X_scaled, rfm_clean, k=5):
    """Fit KMeans + alternate methods. Seed fixed; no UI control."""
    seed = 42
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed)
    clusters = kmeans.fit_predict(X_scaled)
    rfm_k = rfm_clean.copy()
    rfm_k["Cluster"] = clusters

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    agg_labels = agg.fit_predict(X_scaled)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    gmm_labels = gmm.fit_predict(X_scaled)

    s_km = silhouette_score(X_scaled, clusters)
    s_agg = silhouette_score(X_scaled, agg_labels)
    s_gmm = silhouette_score(X_scaled, gmm_labels)
    ch_km = calinski_harabasz_score(X_scaled, clusters)
    db_km = davies_bouldin_score(X_scaled, clusters)
    metrics = {
        "silhouette": {"KMeans": s_km, "Agglomerative": s_agg, "GMM": s_gmm},
        "calinski_harabasz": ch_km,
        "davies_bouldin": db_km,
        "ari": {
            "KM_vs_AGG": adjusted_rand_score(clusters, agg_labels),
            "KM_vs_GMM": adjusted_rand_score(clusters, gmm_labels),
            "AGG_vs_GMM": adjusted_rand_score(agg_labels, gmm_labels)
        }
    }
    return kmeans, clusters, agg_labels, gmm_labels, rfm_k, metrics


# ----------------------- Market Basket helpers --------------------------------
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
st.sidebar.title("CIS 9660  ·  Data Mining  ·  Project 2")

# Keep only model and ROI knobs (no file paths, no product column, no seed)
k_value    = st.sidebar.slider("Number of clusters (K)", 2, 10, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("Rule thresholds")
min_support = st.sidebar.slider("How common (support)", 0.01, 0.10, 0.02, 0.01)
min_conf    = st.sidebar.slider("How reliable (confidence)", 0.10, 0.90, 0.30, 0.05)
min_lift    = st.sidebar.slider("Strength (lift)", 1.0, 3.0, 1.10, 0.05)

st.sidebar.markdown("---")
eligible = st.sidebar.number_input("Eligible baskets for ROI", value=10_000, step=1000)
margin   = st.sidebar.number_input("Gross margin (0-1)", value=0.35, step=0.05, min_value=0.0, max_value=1.0)
cost     = st.sidebar.number_input("Campaign cost ($)", value=1500.0, step=100.0)


# ------------------------------ Load data ------------------------------------
DEFAULT_CSV = "synthetic_retail_transactions.csv"
if not os.path.exists(DEFAULT_CSV):
    st.error(f"CSV not found at {DEFAULT_CSV}. Add the file to your repo.")
    st.stop()

df = load_data(DEFAULT_CSV)
try:
    PRODUCT_COL = ensure_product_col(df)
except ValueError as e:
    st.error(str(e)); st.stop()


# --------------------------- Compute RFM & Clusters ---------------------------
rfm = build_rfm_extras(df)
X_clean, rfm_clean, scaler, X_scaled = scale_and_filter(rfm)
kmeans, clusters, agg_labels, gmm_labels, rfm_k, metrics = fit_kmeans_alt(X_scaled, rfm_clean, k=k_value)

# PCA for visuals
pca = PCA(n_components=3).fit(X_scaled)
X_pca = pca.transform(X_scaled)
exp_var = pca.explained_variance_ratio_

# Build initial market-basket rules once so tabs share them
basket_bool, PRODUCT_COL = build_boolean_basket(df, PRODUCT_COL)
rules_all, itemsets = mine_rules(basket_bool, min_support=min_support, min_conf=min_conf)


# ------------------------------- Tabs ----------------------------------------
tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Market Basket Analysis", "BI Summary and ROI"])


# ================================ TAB 1 ======================================
with tab1:
    st.subheader("Customer Segmentation")

    colA, colB = st.columns([3,2])
    with colA:
        st.markdown("2D PCA Scatter")
        df_plot = pd.DataFrame({
            "PC1": X_pca[:,0], "PC2": X_pca[:,1],
            "Cluster": clusters.astype(int), "CustomerID": rfm_clean["CustomerID"].values
        })
        fig2d = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster",
                           hover_data=["CustomerID"],
                           title=f"2D PCA (variance explained {exp_var[0]+exp_var[1]:.1%})")
        st.plotly_chart(fig2d, use_container_width=True)

        st.markdown("3D PCA Scatter")
        df_plot3 = pd.DataFrame({
            "PC1": X_pca[:,0],
            "PC2": X_pca[:,1],
            "PC3": X_pca[:,2] if X_pca.shape[1] > 2 else X_pca[:,1]*0,
            "Cluster": clusters.astype(int),
            "CustomerID": rfm_clean["CustomerID"].values
        })
        fig3d = px.scatter_3d(df_plot3, x="PC1", y="PC2", z="PC3", color="Cluster",
                              hover_data=["CustomerID"],
                              title=f"3D PCA (variance explained {exp_var[:3].sum():.1%})")
        st.plotly_chart(fig3d, use_container_width=True)

    with colB:
        st.markdown("Segment Statistics")
        sizes = rfm_k["Cluster"].value_counts().sort_index()
        aov = rfm_k.groupby("Cluster")["AvgOrderValue"].mean().round(2)
        mon = rfm_k.groupby("Cluster")["Monetary"].mean().round(2)
        stats_df = pd.DataFrame({"Customers": sizes, "AvgOrderValue": aov, "AvgMonetary": mon})
        st.dataframe(stats_df)

        st.markdown("Validation Snapshots")
        st.write(f"Silhouette — KM: {metrics['silhouette']['KMeans']:.3f}, "
                 f"AGG: {metrics['silhouette']['Agglomerative']:.3f}, "
                 f"GMM: {metrics['silhouette']['GMM']:.3f}")
        st.write(f"Calinski-Harabasz (KM): {metrics['calinski_harabasz']:.1f}")
        st.write(f"Davies-Bouldin (KM): {metrics['davies_bouldin']:.3f}")
        st.write(f"ARI — KM vs AGG: {metrics['ari']['KM_vs_AGG']:.3f}, "
                 f"KM vs GMM: {metrics['ari']['KM_vs_GMM']:.3f}")

    st.markdown("---")
    st.markdown("Segment Comparison (cluster means)")
    cluster_profiles = rfm_k.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
    st.dataframe(cluster_profiles, use_container_width=True)

    st.markdown("Customer Lookup")
    lookup_id = st.text_input("Enter CustomerID")
    if lookup_id:
        try:
            cid = int(lookup_id)
        except:
            cid = lookup_id
        row = rfm_k.loc[rfm_k["CustomerID"] == cid]
        if row.empty:
            st.warning("CustomerID not found in the filtered set.")
        else:
            cl = int(row["Cluster"].iloc[0])
            st.success(f"Customer {cid} is in Cluster {cl}.")
            st.write(row[["Recency","Frequency","Monetary","AvgOrderValue","ProductDiversity",
                          "TotalQuantity","AvgQuantityPerTransaction","CategoryDiversity",
                          "PurchaseSpan","AvgDaysBetweenPurchases"]])


# ================================ TAB 2 ======================================
with tab2:
    st.subheader("Market Basket Analysis")

    # ---------------- Simple recommender ----------------
    st.markdown("Product Recommender")
    all_products = list(basket_bool.columns)
    base_product = st.selectbox("Pick a product", options=sorted(all_products))
    if st.button("Get recommendations"):
        # use all rules; user can further refine below in the explorer
        recs = recommend_from_rules({base_product}, rules_all, top_n=10)
        if recs.empty:
            st.info("No recommendations found for that product.")
        else:
            st.dataframe(recs, use_container_width=True)

    st.markdown("---")

    # ---------------- Rules explorer (friendlier labels) ----------------
    st.markdown("Rule Explorer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_sup_ui = st.number_input("How common (support)", value=float(min_support), step=0.01, min_value=0.0)
    with c2:
        min_conf_ui = st.number_input("How reliable (confidence)", value=float(min_conf), step=0.05, min_value=0.0, max_value=1.0)
    with c3:
        min_lift_ui = st.number_input("Strength (lift)", value=float(min_lift), step=0.05, min_value=1.0)
    with c4:
        max_ant = st.number_input("Max items in the if-part", value=3, step=1, min_value=1)

    include_items = st.multiselect("Must include in the if-part", options=sorted(all_products), default=[])
    exclude_items = st.multiselect("Exclude anywhere", options=sorted(all_products), default=[])

    rules_view = filter_rules(
        rules_all,
        include=include_items or None,
        exclude=exclude_items or None,
        min_support=min_sup_ui, min_conf=min_conf_ui, min_lift=min_lift_ui,
        max_antecedent_len=max_ant
    )
    st.caption(f"{len(rules_view)} rules after filtering.")
    st.dataframe(rules_view[["antecedents","consequents","support","confidence","lift","leverage","conviction"]].head(300),
                 use_container_width=True)

    # make available to tab3
    st.session_state["rules_view"] = rules_view

    st.markdown("---")
    st.markdown("Visualizations")

    # Scatter: support vs confidence, size = lift
    if not rules_view.empty:
        r = rules_view.copy().head(500)
        fig_sc = px.scatter(
            r, x="support", y="confidence", size="lift",
            hover_data=["antecedents","consequents"],
            title="Support vs Confidence (bubble size = Lift)"
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("No rules to plot for current filters.")

    colV1, colV2 = st.columns(2)
    with colV1:
        if not rules_view.empty and HAS_NX:
            st.markdown("Network (top by lift)")
            draw_rules_network_matplotlib(rules_view, top_n=20)
        else:
            st.info("Install networkx for the network plot or relax filters to get rules.")

    with colV2:
        if not rules_view.empty:
            st.markdown("Top Rules by Lift")
            top_rules = rules_view.sort_values(["lift","confidence"], ascending=False).head(10).copy()
            top_rules["rule"] = top_rules.apply(
                lambda r: " + ".join(r["antecedents"]) + " → " + " + ".join(r["consequents"]), axis=1
            )
            fig_bar = go.Figure(
                data=go.Bar(
                    x=top_rules["lift"],
                    y=top_rules["rule"],
                    orientation="h",
                    text=[f"lift {v:.2f}" for v in top_rules["lift"]],
                    textposition="outside"
                )
            )
            fig_bar.update_layout(
                xaxis_title="Lift",
                yaxis_title="Rule",
                height=max(420, 26*len(top_rules)+120),
                margin=dict(l=10, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No rules available for bar chart.")


# ================================ TAB 3 ======================================
with tab3:
    st.subheader("Business Intelligence Summary")

    # Executive summary
    sizes = rfm_k["Cluster"].value_counts().sort_index()
    mon = rfm_k.groupby("Cluster")["Monetary"].mean().round(2)

    st.markdown("Executive Summary")
    st.write(f"- Chosen K: {k_value} (K-Means primary).")
    st.write(f"- Silhouette (K-Means): {metrics['silhouette']['KMeans']:.3f}; "
             f"ARI (KM vs AGG): {metrics['ari']['KM_vs_AGG']:.3f}; "
             f"ARI (KM vs GMM): {metrics['ari']['KM_vs_GMM']:.3f}.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Customer count by cluster")
        st.bar_chart(sizes)
    with col2:
        st.markdown("Average spend per customer by cluster ($)")
        st.bar_chart(mon)

    # Rules for BI
    rules_for_bi = st.session_state.get("rules_view", rules_all)
    top_rules_bi = rules_for_bi.sort_values(["lift","confidence"], ascending=False).head(10).copy()

    st.markdown("Top Rules (by lift)")
    if not top_rules_bi.empty:
        top_rules_bi["rule"] = top_rules_bi.apply(
            lambda r: " + ".join(r["antecedents"]) + " → " + " + ".join(r["consequents"]), axis=1
        )
        st.dataframe(top_rules_bi[["rule","support","confidence","lift","leverage","conviction"]],
                     use_container_width=True)

        # nicer bar
        fig_bar2 = go.Figure(
            data=go.Bar(
                x=top_rules_bi["lift"],
                y=top_rules_bi["rule"],
                orientation="h",
                text=[f"{v:.2f}" for v in top_rules_bi["lift"]],
                textposition="outside"
            )
        )
        fig_bar2.update_layout(
            title="Top Rules by Lift",
            xaxis_title="Lift",
            yaxis_title="Rule",
            height=max(420, 26*len(top_rules_bi)+120),
            margin=dict(l=10, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_bar2, use_container_width=True)
    else:
        st.info("No rules to display.")

    st.markdown("ROI Projections")
    roi_table = build_roi_table(rules_for_bi, df, PRODUCT_COL,
                                eligible=eligible, margin=margin, cost=cost, top_n=10)
    if roi_table.empty:
        st.info("No ROI rows (no rules).")
    else:
        st.dataframe(roi_table.round({
            "support":3, "confidence":3, "lift":2, "base_attach_%":2, "incremental_attach_%":2,
            "avg_consequent_price":2, "expected_incremental_revenue":2, "ROI":2
        }), use_container_width=True)

        # ROI bar
        fig_roi = px.bar(
            roi_table.sort_values("expected_incremental_revenue", ascending=False),
            x="expected_incremental_revenue", y="consequents",
            orientation="h", title="Expected Incremental Revenue by Rule",
            text=roi_table.sort_values("expected_incremental_revenue", ascending=False)["expected_incremental_revenue"].map(lambda v: f"${v:,.0f}")
        )
        fig_roi.update_traces(textposition="outside")
        fig_roi.update_layout(
            xaxis_title="Expected Incremental Revenue ($)",
            yaxis_title="Consequent Item(s)",
            height=max(420, 26*len(roi_table)+120),
            margin=dict(l=10, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("Actionable Recommendations")
    st.write("""
- Loyal Frequent Shoppers: early access, limited-time drops, tiered perks; avoid heavy discounts.
- Bulk High-Spend Buyers: volume pricing, subscribe-and-save, replenishment reminders, cart bundles.
- Steady Long-Term Buyers: milestone coupons, buy-again widgets, free-shipping nudges.
- Variety Seekers (Lapsed): win-back with new arrivals and sampler bundles; time-boxed incentives.
- One-Off Low-Spend Buyers: lightweight re-engagement; limit spend if no response after two touches.
- Cross-sell placement: PDP widgets and cart add-ons using rules with lift ≥ 1.5; lift 1.2–1.5 for email only.
""")


