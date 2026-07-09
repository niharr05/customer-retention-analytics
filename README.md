# 🏦 Customer Engagement & Product Utilization Analytics for Retention Strategy

**An end-to-end retail banking analytics platform** that transforms raw customer
data into an executive-ready retention strategy toolkit — combining a
production-grade Python analytics pipeline with an interactive Streamlit
dashboard.

---

## 📌 Project Overview

Retail banks lose significant revenue every year to preventable customer
churn. This project builds a complete analytics solution that identifies
**who is at risk of leaving, why they are at risk, and what the bank can do
about it** — using engagement behavior and product-holding patterns rather
than churn prediction alone.

The project is split into two cooperating components:

| Component | Role |
|---|---|
| `analysis.py` | The analytics engine: validation, cleaning, feature engineering, KPI computation, chart generation, and automated insight generation. |
| `app.py` | A 7-page interactive Streamlit dashboard that consumes `analysis.py` to let stakeholders explore the data live, filter by segment, and export reports. |

---

## 🎯 Business Problem

> *"We know customers are leaving. We don't know which ones matter most,
> or which levers actually move retention."*

Banks typically have plenty of raw transactional and demographic data, but
struggle to convert it into **actionable retention priorities**. This
project addresses three concrete business questions:

1. **Who are our most valuable customers, and are they engaged?**
2. **Which behavioral and product-holding patterns are associated with churn?**
3. **Where should retention budget and relationship-manager time be spent first?**

---

## 🎯 Objectives

- Build a reusable, auditable analytics pipeline for customer engagement data.
- Engineer business-meaningful features (engagement, relationship strength,
  premium/at-risk flags, retention scoring) beyond raw columns.
- Surface **20+ professional visualizations** covering demographic, behavioral,
  and product dimensions of churn.
- Automatically generate **plain-English, data-grounded business insights**.
- Deliver an interactive dashboard that lets non-technical stakeholders filter,
  search, and export findings without touching code.

---

## 📊 Dataset

The pipeline expects a retail banking customer CSV (commonly known as the
"Churn Modelling" dataset structure) with columns such as:

| Column | Description |
|---|---|
| `Year` | Record year |
| `CustomerId` | Unique customer identifier |
| `Surname` | Customer surname |
| `CreditScore` | Customer credit score |
| `Geography` | Customer's country/region |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer holds a credit card (0/1) |
| `IsActiveMember` | Whether the customer is an active member (0/1) |
| `EstimatedSalary` | Customer's estimated salary |
| `Exited` | Churn flag (1 = churned, 0 = retained) |

> **Note:** Column detection is automatic and case/format-insensitive — if
> your source file uses slightly different names (e.g. `Churn` instead of
> `Exited`, or `Country` instead of `Geography`), the pipeline will still
> correctly map them.

Place your CSV file (e.g. `European_Bank.csv`) in the project root directory
before running the pipeline.

---

## 🏗️ Architecture

```
Customer_Engagement_Retention_Analytics/
│
├── analysis.py              # Full analytics pipeline (validation → cleaning →
│                             # feature engineering → KPIs → charts → insights)
├── app.py                   # 7-page Streamlit dashboard (imports analysis.py)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── Executive_Summary.md     # Government & Regulatory Briefing summary of key findings
├── charts/                  # 20 auto-generated PNG charts
└── processed_data/          # Cleaned data + KPI + business summary exports
    ├── cleaned_customer_data.csv
    ├── kpi_report.csv
    └── business_summary.csv
```

**Design principle:** `app.py` never re-implements business logic. Every KPI,
feature, and chart definition lives in `analysis.py` and is imported by the
dashboard, keeping a single source of truth for the analytics.

---

## ⚙️ Installation

```bash
# 1. Clone or download this project folder
cd Customer_Engagement_Retention_Analytics

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset CSV in this folder (e.g. European_Bank.csv)
```

---

## ▶️ Usage

### Run the full analytics pipeline (generates charts + processed data)

```bash
python analysis.py
```

This will:
- Validate and clean the raw dataset
- Engineer all business features
- Compute 17 executive KPIs
- Generate and save 20 charts to `charts/`
- Print automatically generated business insights
- Export `cleaned_customer_data.csv`, `kpi_report.csv`, and
  `business_summary.csv` to `processed_data/`

### Launch the interactive dashboard

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

---

## ✨ Features

- **Automatic column detection** — resilient to minor naming differences in source data.
- **Full data-quality audit** — missing values, duplicates, invalid binaries, impossible ages, negative balances/salaries, invalid product counts.
- **Intelligent cleaning** — median/mode imputation, IQR-based outlier capping (not deletion), datatype correction.
- **12 engineered features** — Engagement Score, Relationship Strength Index, Product Depth Index, Premium/Sticky/At-Risk-Premium flags, Lifetime Segment, Age/Balance/Salary categories, heuristic Risk Level, and a composite Retention Score.
- **17 executive KPIs** — churn/retention rates, product utilization index, engagement-retention ratio, high-balance disengagement rate, credit card stickiness score, and more.
- **20 publication-quality charts** — Plotly (interactive) and Matplotlib (static), covering demographics, product behavior, engagement, and an executive multi-panel dashboard figure.
- **Automated business insights** — plain-English, statistically grounded narrative insights generated directly from the data (including explicit small-sample caveats where relevant).
- **7-page Streamlit dashboard** — Executive Dashboard, Customer Analysis, Engagement Analytics, Product Utilization, Premium Customer Detector, Retention Analytics, and Business Insights.
- **Full interactivity** — sidebar filters (country, gender, active status, and sliders for age, balance, salary, and products), customer search, sortable data tables, CSV downloads, and dynamic risk alerts.

---

## 📷 Screenshots

The pipeline generates 20 publication-quality charts. Example outputs:
- **`charts/20_executive_kpi_dashboard.png`** - Multi-panel executive summary overview.
- **`charts/11_correlation_heatmap.png`** - Relationships between numeric features.
- **`charts/18_salary_vs_balance_scatter.png`** - Salary vs Balance distribution by Churn.
*(All charts are available in the `charts/` directory after running the pipeline.)*

---

## 🔑 Key Metrics (KPIs)

The pipeline calculates 17 distinct KPIs, including:
| Metric | Description |
|---|---|
| **Total Customers** | Number of customers in the current filtered view |
| **Churn Rate (%)** | Percentage of customers who have exited |
| **Product Utilization Index** | Average number of products held relative to the maximum (4) |
| **Engagement Retention Ratio** | Average engagement of retained customers vs churned customers |
| **High Balance Disengagement Rate** | Percentage of high-balance customers who are inactive |
| **Credit Card Stickiness Score** | Retention rate among credit card holders |

---

## ⚙️ Technical Highlights

- **Column Auto-Detection:** The pipeline handles variations in dataset headers via alias mapping.
- **Intelligent Data Cleaning:** Rather than deleting outlier rows, it clips impossible values and caps extreme outliers using the IQR method. This preserves precious sample size, especially for legitimate high-net-worth customer records.
- **Composite Scoring Methodology:** Features like Engagement Score and Relationship Strength Index are calculated using domain-relevant weighted combinations of underlying data points.
- **Robust Testing:** A full `pytest` suite is included to verify data loading, validation, cleaning, feature engineering, KPI computations, and insights generation.

---

## 💡 Business Insights (Sample Output)

The pipeline automatically generates insights such as:

- Customers holding 2 products churn the least, while customers with 3+
  products churn sharply more — suggesting aggressive product cross-selling
  can itself become a churn signal.
- High-value ("Premium") but inactive/under-engaged customers represent a
  measurable hidden revenue risk requiring proactive relationship management.
- High account balance does **not** guarantee loyalty — high-balance
  customers in this portfolio churn at a higher rate than the overall base.
- Inactive members churn at roughly double the rate of active members,
  making reactivation campaigns a high-leverage retention lever.
- Regional churn rates vary meaningfully by geography, pointing to
  market-specific service or competitive dynamics worth investigating.

*(Full, current-run insights are available in `processed_data/business_summary.csv`
and on the "Business Insights" page of the dashboard.)*

---

## 🚀 Future Improvements

- Add a trained machine learning propensity-to-churn model (e.g. XGBoost or LightGBM) alongside the current heuristic Risk Level, with SHAP values for explainability.
- Incorporate transactional and behavioral time-series data to detect trend-based early-warning signals, rather than relying solely on snapshot-based point-in-time scoring.
- Implement cohort and survival analysis (time-to-churn) views to better understand customer lifetime value.
- Persist dashboard filter state and support role-based views (e.g., branch manager vs. regional executive).
- Containerize the entire dashboard application and analytics pipeline using Docker for one-command deployment and environment reproducibility.

---

## 👤 Author

Built by a Senior Data Science & Analytics Engineering team as a
demonstration-grade banking retention analytics platform, designed to be
readable, extensible, and presentation-ready for recruiters, evaluators,
and banking stakeholders alike.
