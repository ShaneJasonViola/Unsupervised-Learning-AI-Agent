Retail Analytics — Customer Segmentation & Market Basket
========================================================

This repo contains two pieces:

1) **analysis_pipeline.py** — a notebook‑friendly script that builds customer segments (RFM + extras), compares clustering methods, mines association rules with Apriori, and outputs clear visuals plus an ROI table.
2) **app_streamlit.py** — an interactive Streamlit app that lets you explore the segments, play with market‑basket rules, and view a business summary with recommendations and ROI.

If you’re skimming: install the requirements, put your CSV next to these files (or keep the synthetic sample), run the analysis once, then launch Streamlit.

---

Quick Start (local)
-------------------

### 1) Install dependencies

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Put your data in place

The code looks for a file named **`synthetic_retail_transactions.csv`** by default.  
If you have your own data, you can either rename it to that or change the path inside the files.

**Expected columns** (typical retail schema):
- `InvoiceNo` (transaction id)
- `CustomerID`
- `InvoiceDate` (parseable dates)
- `ProductID` and/or `ProductName` / `Description`
- `Category` (optional, but used for “category diversity”)
- `UnitPrice`
- `Quantity`
- `Amount` (optional — if missing, it’s computed as `Quantity * UnitPrice`).

### 3) Run the analysis once (to generate charts & the ROI table)

```bash
python analysis_pipeline.py
```

> Tip: the analysis file is notebook‑friendly (no command‑line args). If you’re using your own CSV, open the file and change the default path at the very top, then run it again.

The script will:
- Clean the data and build **RFM + behavioral features** (AOV, diversity, span, gaps).
- Fit **K‑Means** (primary), plus **Agglomerative** and **GMM** for comparison.
- Plot the **elbow** and **silhouette** guidance, 2D PCA scatter, cluster profiles, etc.
- Run **Apriori** to get frequent itemsets and **association rules**.
- Print a **top‑rules** summary and write **`roi_rules_table.csv`** to disk.
- Create a few report‑ready charts (cluster sizes, AOV, rule strength, ROI bars).

### 4) Launch the Streamlit app

```bash
streamlit run app_streamlit.py
```

What you’ll see in the app:

- **Customer Segmentation**
  - 2D and 3D cluster plots (PCA).
  - A clean table comparing segments on the key features.
  - A **Customer Lookup** box — type a `CustomerID` and see which segment they’re in.
  - Simple segment stats (counts and spend).

- **Market Basket**
  - A simpler rules explorer with just the useful sliders (support, confidence, lift).
  - A **Product Recommender** — pick a product and get suggested add‑ons.
  - Optional **network graph** (if `networkx` is installed) and a parallel‑coordinates view.
  - A scatter for support vs confidence (bubble size = lift).

- **BI Summary**
  - Short executive summary (K used, validation snapshots).
  - Top rules table and **ROI** bar chart using your chosen thresholds.
  - Actionable bullet points by segment.

---

Run in Google Colab
-------------------

You can run both the analysis and the Streamlit app from Colab.

### 1) Open Colab and install packages

```python
!pip -q install --upgrade pip
!pip -q install -r https://raw.githubusercontent.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/main/requirements.txt
```

> If you don’t have the repo public yet, you can install packages one by one:
>
> ```python
> !pip -q install numpy pandas matplotlib seaborn scikit-learn mlxtend plotly streamlit networkx pyngrok
> ```

### 2) Get your data and code into Colab

**Option A — Clone the GitHub repo** (easiest once this project is on GitHub):

```python
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
%cd <YOUR_REPO_NAME>
```

**Option B — Upload files manually**: use the Colab file pane (left sidebar) to upload
`analysis_pipeline.py`, `app_streamlit.py`, and your CSV.

### 3) Run the analysis

```python
!python analysis_pipeline.py
```

This will generate charts inline (in Colab logs) and write `roi_rules_table.csv` to the working directory.

### 4) (Optional) Launch the Streamlit app from Colab

Colab isn’t built for long‑running web apps, but you can tunnel the app for quick demos using **pyngrok**.

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
public_url
```

Then, in another cell:

```python
!streamlit run app_streamlit.py --server.headless true --server.port 8501
```

Open the URL printed by the `ngrok.connect` cell. When you’re done, **Interrupt execution** to stop the app.

> Note: ngrok URLs are temporary and will change each session. Colab must stay running for the link to work.

---

Project Structure
-----------------

```
.
├── analysis_pipeline.py         # RFM features, clustering, Apriori, charts, ROI export
├── app_streamlit.py             # Streamlit UI (segments, rules, recommender, BI summary)
├── requirements.txt             # Python dependencies
├── synthetic_retail_transactions.csv  # sample dataset (or place your own)
└── README.md / README.txt
```

The app automatically detects the product column from `ProductName`, `Description`, or `ProductID` (in that order).
It also computes `Amount` if it’s missing.

---

Troubleshooting
---------------

- **“ModuleNotFoundError: mlxtend”**  
  Re‑install dependencies: `pip install -r requirements.txt`. In Colab, run the `pip` cells again.

- **“CSV not found” in Streamlit**  
  Make sure `synthetic_retail_transactions.csv` (or your file with that name) is in the same folder as `app_streamlit.py`.
  The app does not show an upload widget by design.

- **“No product column found”**  
  The app tries `ProductName`, `Description`, then `ProductID`. If none exist, add one of those columns.

- **Network graph doesn’t show**  
  The network view needs `networkx`. It’s in `requirements.txt`, but if you removed it, reinstall with `pip install networkx`.

- **Colab kills my app**  
  That happens when the runtime sleeps or if the notebook is interrupted. Use it for quick demos only; for a stable URL, deploy on Streamlit Community Cloud.

---

Deployment (Streamlit Community Cloud)
--------------------------------------

1. Push your repo to GitHub.
2. Go to **share.streamlit.io**, sign in with GitHub, and pick your repo and `app_streamlit.py` as the entry point.
3. Set Python version to **3.10+** and point to **requirements.txt**.
4. Click **Deploy**. You’ll get a public URL you can share.

---

License & Credits
-----------------

Use this freely for coursework and internal demos. If you build on it for another project, a small attribution is appreciated.

Questions or stuck? Open an issue and paste any error messages — I’ll help you debug.



