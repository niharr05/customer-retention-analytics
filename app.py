"""
==============================================================================
 CUSTOMER ENGAGEMENT & PRODUCT UTILIZATION ANALYTICS - STREAMLIT DASHBOARD
==============================================================================

A 7-page interactive Streamlit dashboard for exploring customer engagement,
product utilization, and retention analytics in a retail banking context.

Pages
-----
1. Executive Dashboard   - Portfolio health KPIs with risk alerts
2. Customer Analysis     - Demographics, search, and data table
3. Engagement Analytics  - Composite engagement & relationship scoring
4. Product Utilization   - Product-holding patterns vs churn
5. Premium Customer Det. - High-value customer identification & risk
6. Retention Analytics   - Retention scoring and recommendations
7. Business Insights     - Auto-generated insights with export

Usage
-----
    $ streamlit run app.py

Dependencies
------------
All business logic is imported from analysis.py — this file handles only
the presentation layer (layout, charts, filters, styling).
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as az  # All data processing / feature engineering / KPI logic

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional enterprise color system for dashboard UI consistency.
APP_COLORS = {
    "primary": "#0B3C5D",      # deep navy
    "accent": "#1F7A8C",       # executive teal
    "success": "#2E8B57",      # balanced green
    "danger": "#B42318",       # high-contrast red
    "warning": "#D97706",      # amber warning
    "neutral": "#D5DCE5",      # muted border
    "surface": "#FFFFFF",
    "bg": "#F3F6FA",
    "text": "#0F172A",
    "text_muted": "#5B6B7F",
    "sidebar_bg": "#0A2540",
}

PRIMARY = APP_COLORS["primary"]
SUCCESS = APP_COLORS["success"]
DANGER = APP_COLORS["danger"]
WARNING = APP_COLORS["warning"]
ACCENT = APP_COLORS["accent"]
NEUTRAL = APP_COLORS["neutral"]
CHURN_COLOR_MAP = {"Retained": SUCCESS, "Churned": DANGER}

REQUIRED_ENGINEERED_COLUMNS = {
    "CustomerId", "Surname", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
    "EngagementScore", "RelationshipStrengthIndex", "RetentionScore", "PremiumCustomer",
    "AtRiskPremiumCustomer", "StickyCustomer", "RiskLevel", "CustomerLifetimeSegment",
    "ProductDepthIndex", "EngagementTier", "SalaryBalanceMismatch", "SalaryBalanceMismatchType",
    "EngagementProfile", "RetentionStabilityTier",
}


def _resolve_source_dataset() -> str:
    """Identify which raw CSV source is present in project root."""
    for candidate in az.RAW_CSV_CANDIDATES:
        candidate_path = os.path.join(az.BASE_DIR, candidate)
        if os.path.exists(candidate_path):
            return candidate
    return "Unknown source"


def _safe_churn_label(series: pd.Series) -> pd.Series:
    """Avoid dependency on private module methods for churn label mapping."""
    return series.map({1: "Churned", 0: "Retained"}).fillna("Unknown")

# --------------------------------------------------------------------------
# CUSTOM CSS - PROFESSIONAL BANKING THEME
# --------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{
        --primary: {PRIMARY};
        --secondary: #3B556E;
        --accent: {ACCENT};
        --success: {SUCCESS};
        --danger: {DANGER};
        --warning: {WARNING};
        --neutral: {NEUTRAL};
        --surface: {APP_COLORS["surface"]};
        --bg: {APP_COLORS["bg"]};
        --border: {APP_COLORS["neutral"]};
        --text: {APP_COLORS["text"]};
        --text-muted: {APP_COLORS["text_muted"]};
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── Base ────────────────────────────────────────────── */
    .main {{
        background:
            radial-gradient(1100px 600px at 8% -10%, rgba(31, 122, 140, 0.09), transparent 65%),
            radial-gradient(950px 500px at 92% -18%, rgba(11, 60, 93, 0.10), transparent 70%),
            var(--bg);
    }}
    .stApp {{
        font-family: 'Inter', 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text);
    }}
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        gap: 32px;
    }}
    h1 {{
        color: var(--text);
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.03em;
    }}
    h2, h3 {{
        color: var(--text);
        font-size: 22px;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}

    /* ── Hero / Page Header ──────────────────────────────── */
    .hero-section {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .hero-section:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }}
    .pill {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: #EAF3FD;
        color: var(--primary);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }}
    .hero-title {{
        font-size: 32px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.5rem;
        line-height: 1.2;
        letter-spacing: -0.04em;
    }}
    .hero-subtitle {{
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }}

    /* ── Metric Cards (native st.metric) ─────────────────── */
    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }}
    div[data-testid="stMetricLabel"] {{
        color: var(--text-muted);
        font-weight: 500;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--text);
        font-weight: 800;
        font-size: 40px;
        letter-spacing: -0.02em;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {APP_COLORS["sidebar_bg"]};
        border-right: none;
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] * {{
        color: #EAF2FB !important;
    }}
    section[data-testid="stSidebar"] label {{
        color: #C7D6E8 !important;
        font-weight: 500;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 4px;
    }}
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stMultiSelect > div,
    section[data-testid="stSidebar"] .stTextInput > div,
    section[data-testid="stSidebar"] .stNumberInput > div,
    section[data-testid="stSidebar"] .stSlider > div {{
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 8px 12px;
        transition: border 0.2s ease, background 0.2s ease;
    }}
    section[data-testid="stSidebar"] .stSelectbox > div:hover,
    section[data-testid="stSidebar"] .stMultiSelect > div:hover {{
        background: rgba(255,255,255,0.12);
        border-color: rgba(255,255,255,0.32);
    }}
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {{
        background: rgba(10, 37, 64, 0.75) !important;
        border-color: rgba(199, 214, 232, 0.35) !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input,
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSlider p {{
        color: #F8FCFF !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] [role="listbox"] * {{
        color: #0F172A !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="tag"] {{
        background: #1F7A8C !important;
        border: 1px solid #47A8BC !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="tag"] span,
    section[data-testid="stSidebar"] [data-baseweb="tag"] svg {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div {{
        background: rgba(31, 122, 140, 0.65) !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(11, 60, 93, 0.45);
    }}
    .sidebar-panel {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px;
        margin: 8px 8px 16px 8px;
    }}
    .filter-section-title {{
        color: #F9FAFB;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.01em;
    }}
    .filter-note {{
        color: rgba(226,232,240,0.7);
        font-size: 0.85rem;
        margin-top: 8px;
    }}

    /* ── Content Cards ───────────────────────────────────── */
    .insight-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px -1px rgba(0,0,0,0.03);
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.5;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .insight-card:hover {{
        transform: translateX(4px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }}
    .risk-alert {{
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 4px solid var(--danger);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        color: #991B1B;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    .recommendation-card {{
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 4px solid var(--success);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        color: #166534;
        font-weight: 600;
    }}

    /* ── Custom Metric Cards (HTML) ──────────────────────── */
    .metric-card {{
        background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        min-height: 110px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 16px;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }}
    .metric-label {{
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }}
    .metric-value {{
        color: var(--text);
        font-size: 40px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }}
    .metric-delta {{
        color: var(--success);
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }}

    /* ── Chart Containers ────────────────────────────────── */
    .chart-frame {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 16px;
    }}
    .chart-frame:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }}

    /* ── Segmented Nav / Tabs ────────────────────────────── */
    .stRadio > div {{
        background: #F1F5F9;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 4px;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }}
    .stRadio > div label {{
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 0;
        transition: all 0.2s ease;
        cursor: pointer;
        color: var(--text-muted);
        font-weight: 500;
        flex: 1;
        text-align: center;
    }}
    .stRadio > div label:hover {{
        background: #E2E8F0;
        color: var(--text);
    }}
    .stRadio input[type="radio"] {{
        display: none;
    }}
    .stRadio input[type="radio"]:checked + label,
    .stRadio input[type="radio"]:checked + div + label,
    .stRadio input[type="radio"]:checked + label > div {{
        background: var(--surface) !important;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        font-weight: 600 !important;
    }}

    /* ── Native Streamlit Accent Controls ───────────────── */
    .stTabs [role="tablist"] {{
        border-bottom: 1px solid var(--border);
        gap: 0.4rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border-radius: 14px;
        padding: 0.35rem 0.4rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    }}
    .stTabs [role="tab"] {{
        color: var(--text-muted) !important;
        border-radius: 10px;
        font-size: 15px !important;
        font-weight: 500 !important;
        min-height: 40px;
        padding: 0.4rem 0.9rem !important;
        transition: transform 0.22s ease, background-color 0.22s ease, color 0.22s ease, box-shadow 0.22s ease;
        position: relative;
    }}
    .stTabs [role="tab"]:hover {{
        color: var(--primary) !important;
        background: rgba(31, 122, 140, 0.08);
        transform: translateY(-1px);
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        color: var(--primary) !important;
        background: rgba(31, 122, 140, 0.12);
        box-shadow: 0 6px 14px rgba(31, 122, 140, 0.18);
        transform: translateY(-1px);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        height: 2px !important;
        border-radius: 999px;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox [data-baseweb="select"]:focus-within,
    .stMultiSelect [data-baseweb="select"]:focus-within {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    /* ── Table Typography ───────────────────────────────── */
    .stDataFrame [role="columnheader"],
    .stDataFrame [role="gridcell"],
    .stTable th,
    .stTable td {{
        font-size: 14px !important;
        font-weight: 400 !important;
        line-height: 1.45 !important;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {{
        transition: all 0.2s ease;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface);
    }}
    .stButton > button:hover {{
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-color: #CBD5E1;
        transform: translateY(-1px);
    }}

    /* ── Footer ──────────────────────────────────────────── */
    footer {{visibility: hidden;}}
    .app-footer {{
        text-align: center;
        color: var(--text-muted);
        padding: 32px 0 16px 0;
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
        margin-top: 48px;
        font-weight: 500;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# Theme CSS is injected from the sidebar selector earlier; no-op placeholder.


# ==========================================================================
# DATA LOADING (CACHED)
# ==========================================================================

@st.cache_data(show_spinner=False)
def get_pipeline_outputs() -> tuple[pd.DataFrame, dict[str, float | int], list[str]]:
    """Run (or load cached results of) the full analytics pipeline.

    Uses previously exported processed data if available to avoid
    recomputation on every dashboard reload; otherwise runs the pipeline
    fresh via analysis.py.

    Returns
    -------
    tuple[pd.DataFrame, dict, list]
        (engineered dataframe, kpi dictionary, insight list)
    """
    cleaned_path = os.path.join(az.DATA_DIR, "cleaned_customer_data.csv")

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        missing_cols = REQUIRED_ENGINEERED_COLUMNS - set(df.columns)
        if missing_cols:
            # Fall back to full pipeline if cached export is stale/incomplete.
            df, kpis, insights = az.run_full_pipeline()
            return df, kpis, insights

        # Recompute KPIs & insights fresh from the cached cleaned/engineered data
        # (fast, no re-validation/re-cleaning needed since data is already clean).
        kpis = az.calculate_kpis(df)
        insights = az.generate_business_insights(df, kpis)
    else:
        df, kpis, insights = az.run_full_pipeline()

    return df, kpis, insights


with st.spinner("Loading customer analytics engine..."):
    full_df, base_kpis, base_insights = get_pipeline_outputs()


# ==========================================================================
# SIDEBAR - GLOBAL FILTERS
# ==========================================================================

st.sidebar.title("🏦 Retention Analytics")
st.sidebar.markdown("<div class='filter-section-title'>🔎 <strong>Filters</strong></div>", unsafe_allow_html=True)

countries = sorted(full_df["Geography"].unique().tolist())
selected_countries = st.sidebar.multiselect("Country", countries, default=countries)

genders = sorted(full_df["Gender"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

age_min, age_max = int(full_df["Age"].min()), int(full_df["Age"].max())
selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

prod_min, prod_max = int(full_df["NumOfProducts"].min()), int(full_df["NumOfProducts"].max())
selected_products = st.sidebar.slider("Number of Products", prod_min, prod_max, (prod_min, prod_max))

bal_min, bal_max = float(full_df["Balance"].min()), float(full_df["Balance"].max())
selected_balance = st.sidebar.slider("Balance Range", bal_min, bal_max, (bal_min, bal_max))

sal_min, sal_max = float(full_df["EstimatedSalary"].min()), float(full_df["EstimatedSalary"].max())
selected_salary = st.sidebar.slider("Salary Range", sal_min, sal_max, (sal_min, sal_max))

active_options = ["Active", "Inactive"]
selected_active = st.sidebar.multiselect("Active Status", active_options, default=active_options)

engagement_min, engagement_max = float(full_df["EngagementScore"].min()), float(full_df["EngagementScore"].max())
selected_engagement = st.sidebar.slider(
    "Engagement Score Range",
    engagement_min,
    engagement_max,
    (engagement_min, engagement_max),
)

if "EngagementTier" in full_df.columns:
    engagement_tiers = [tier for tier in ["Low", "Medium", "High"] if tier in full_df["EngagementTier"].dropna().astype(str).unique().tolist()]
    selected_engagement_tiers = st.sidebar.multiselect("Engagement Tier", engagement_tiers, default=engagement_tiers)
else:
    engagement_tiers = []
    selected_engagement_tiers = []
    st.sidebar.caption("Engagement tier data is unavailable in the current dataset.")

st.sidebar.markdown("</div>", unsafe_allow_html=True)

NAV_OPTIONS = [
    "📊 Executive Dashboard",
    "👥 Customer Analysis",
    "💡 Engagement Analytics",
    "📦 Product Utilization",
    "💎 Premium Customer Detector",
    "🔄 Retention Analytics",
    "📈 Business Insights",
]


# --- Apply filters --------------------------------------------------------
active_map = {"Active": 1, "Inactive": 0}
selected_active_vals = [active_map[a] for a in selected_active]

if "EngagementTier" in full_df.columns:
    if selected_engagement_tiers:
        engagement_tier_mask = full_df["EngagementTier"].isin(selected_engagement_tiers)
    else:
        engagement_tier_mask = pd.Series(False, index=full_df.index)
else:
    engagement_tier_mask = pd.Series(True, index=full_df.index)

filtered_df = full_df[
    (full_df["Geography"].isin(selected_countries))
    & (full_df["Gender"].isin(selected_genders))
    & (full_df["Age"].between(*selected_age))
    & (full_df["NumOfProducts"].between(*selected_products))
    & (full_df["Balance"].between(*selected_balance))
    & (full_df["EstimatedSalary"].between(*selected_salary))
    & (full_df["IsActiveMember"].isin(selected_active_vals))
    & (full_df["EngagementScore"].between(*selected_engagement))
    & engagement_tier_mask
].copy()

dataset_label = _resolve_source_dataset()
data_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M")

# --- Filter summary in sidebar -------------------------------------------
filter_pct = len(filtered_df) / len(full_df) * 100 if len(full_df) > 0 else 0
st.sidebar.markdown(
    f'<div style="text-align:center; padding:8px 0 4px 0; '
    f'color:#94A3B8; font-size:0.82rem;">'
    f'Showing <strong style="color:#E2E8F0;">{len(filtered_df):,}</strong> '
    f'of {len(full_df):,} customers ({filter_pct:.0f}%)</div>',
    unsafe_allow_html=True,
)

# --- Dataset info ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div style="font-size:0.78rem; color:#94A3B8; padding:4px 12px;">'
    f'📂 <strong>Dataset:</strong> {len(full_df):,} rows × {full_df.shape[1]} columns<br>'
    f'🗂️ <strong>Source:</strong> {dataset_label}<br>'
    f'🔧 <strong>Engine:</strong> analysis.py pipeline<br>'
    f'🕐 <strong>Refreshed:</strong> {data_refresh_time}</div>',
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No customers match the selected filters. Please broaden your filter selection.")
    st.stop()

filtered_df["ChurnLabel"] = _safe_churn_label(filtered_df["Exited"])
kpis = az.calculate_kpis(filtered_df)
insights = az.generate_business_insights(filtered_df, kpis)


# ==========================================================================
# SHARED UI HELPERS
# ==========================================================================

def render_page_header(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """Render a polished page header for each dashboard section."""
    badge_html = f'<div class="pill">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-section">
            {badge_html}
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpi_specs: Sequence[tuple[str, str, Optional[str]]]) -> None:
    """Render a row of polished metric cards."""
    cols = st.columns(len(kpi_specs))
    for col, (label, value, delta) in zip(cols, kpi_specs):
        delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_insights(insight_list: list[str]) -> None:
    """Render business insights as styled cards."""
    for insight in insight_list:
        st.markdown(f'<div class="insight-card">💡 {insight}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Render the standard application footer."""
    st.markdown(
        '<div class="app-footer">Customer Engagement & Retention Analytics Platform '
        '&nbsp;|&nbsp; Built with Python, Pandas, Plotly & Streamlit '
        '&nbsp;|&nbsp; For internal banking analytics use</div>',
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply a single enterprise plotting style across all visualizations."""
    fig.update_layout(
        template=az.PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Manrope, Segoe UI, sans-serif", "color": APP_COLORS["text"], "size": 12},
        title={"font": {"size": 22, "color": APP_COLORS["text"], "family": "Inter, Manrope, Segoe UI, sans-serif"}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "family": "Inter, Manrope, Segoe UI, sans-serif"},
        },
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF2F7", zeroline=False, tickfont={"size": 12}, title_font={"size": 12})
    fig.update_yaxes(showgrid=True, gridcolor="#EDF2F7", zeroline=False, tickfont={"size": 12}, title_font={"size": 12})
    return fig


def render_plot(fig: go.Figure) -> None:
    """Render a themed plotly chart with consistent config."""
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True, config={"displaylogo": False})


def benchmark_status(actual: float, target: float, lower_is_better: bool = False) -> tuple[str, str, str]:
    """Return benchmark label, semantic color, and direction text for KPI tracking."""
    delta = target - actual if lower_is_better else actual - target
    if delta >= 2:
        return "Ahead", SUCCESS, f"+{abs(delta):.1f} vs target"
    if delta >= -2:
        return "Near Target", WARNING, f"{delta:+.1f} vs target"
    return "Off Track", DANGER, f"{delta:+.1f} vs target"


def render_benchmark_panel() -> None:
    """Show an executive benchmark summary with traffic-light status cues."""
    churn_label, churn_color, churn_delta = benchmark_status(kpis["Churn Rate (%)"], 15.0, lower_is_better=True)
    retention_label, retention_color, retention_delta = benchmark_status(kpis["Retention Rate (%)"], 85.0)
    active_label, active_color, active_delta = benchmark_status(kpis["Active Customer (%)"], 60.0)

    st.markdown("### Benchmark Snapshot")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"<div class='insight-card'><strong>Churn Benchmark (<=15%)</strong><br>"
            f"<span style='color:{churn_color}; font-weight:700;'>{churn_label}</span>"
            f" <span style='color:#64748B;'>({churn_delta})</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='insight-card'><strong>Retention Benchmark (>=85%)</strong><br>"
            f"<span style='color:{retention_color}; font-weight:700;'>{retention_label}</span>"
            f" <span style='color:#64748B;'>({retention_delta})</span></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"<div class='insight-card'><strong>Active Base Benchmark (>=60%)</strong><br>"
            f"<span style='color:{active_color}; font-weight:700;'>{active_label}</span>"
            f" <span style='color:#64748B;'>({active_delta})</span></div>",
            unsafe_allow_html=True,
        )


def get_action_queue() -> list[tuple[str, str, str]]:
    """Generate a prioritized action queue from current filter metrics."""
    actions: list[tuple[str, str, str]] = []

    at_risk_pct = (filtered_df["AtRiskPremiumCustomer"] == "Yes").mean() * 100
    if at_risk_pct >= 5:
        actions.append((
            "P1 - Critical",
            f"Protect premium value pool ({at_risk_pct:.1f}% at-risk premium customers)",
            "Assign relationship managers to proactive outreach in 7 days.",
        ))

    if kpis["High Balance Disengagement Rate (%)"] >= 20:
        actions.append((
            "P1 - Critical",
            f"Reduce high-balance disengagement ({kpis['High Balance Disengagement Rate (%)']:.1f}%)",
            "Run concierge activation journeys for top-balance inactive segment.",
        ))

    if kpis["Active Customer (%)"] < 60:
        actions.append((
            "P2 - High",
            f"Increase active membership ({kpis['Active Customer (%)']:.1f}% current)",
            "Launch app/login reactivation campaign with tenure-based incentives.",
        ))

    if kpis["Product Utilization Index"] < 55:
        actions.append((
            "P2 - High",
            f"Improve product depth ({kpis['Product Utilization Index']:.1f}% utilization)",
            "Offer needs-based bundles anchored on 2-product sweet spot.",
        ))

    if not actions:
        actions.append((
            "P3 - Monitor",
            "No immediate red flags in selected segment",
            "Maintain current playbook and monitor monthly KPI drift.",
        ))
    return actions


def render_action_queue() -> None:
    """Render prioritized retention actions for executive decisioning."""
    st.subheader("Executive Action Queue")
    for priority, issue, action in get_action_queue():
        color = DANGER if "P1" in priority else (WARNING if "P2" in priority else SUCCESS)
        st.markdown(
            f"<div class='insight-card' style='border-left-color:{color};'>"
            f"<strong>{priority}</strong><br>{issue}<br>"
            f"<span style='color:#475569;'>Next Action: {action}</span></div>",
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================================================

def _delta_str(filtered_val: float, base_val: float, suffix: str = "", invert: bool = False) -> Optional[str]:
    """Return a delta indicator string (▲/▼) comparing filtered vs full dataset.

    Parameters
    ----------
    filtered_val : float
        The KPI value for the current filtered view.
    base_val : float
        The KPI value for the unfiltered full dataset.
    suffix : str
        Optional suffix (e.g., '%', ' pts').
    invert : bool
        If True, a negative delta is colored green (e.g., for churn rate).
    """
    if base_val == filtered_val or base_val == 0:
        return None
    diff = filtered_val - base_val
    arrow = "▲" if diff > 0 else "▼"
    color = ("#DC2626" if diff > 0 else "#059669") if invert else ("#059669" if diff > 0 else "#DC2626")
    return f'<span style="color:{color}">{arrow} {abs(diff):.1f}{suffix} vs. all</span>'


def page_executive_dashboard() -> None:
    """Render the top-level executive KPI dashboard page."""
    render_page_header(
        "Executive Dashboard",
        "High-level portfolio health, churn, and engagement overview.",
        "Executive View",
    )

    render_kpi_row([
        ("Total Customers", f"{kpis['Total Customers']:,}",
         _delta_str(kpis['Total Customers'], base_kpis['Total Customers'])),
        ("Churn Rate", f"{kpis['Churn Rate (%)']:.1f}%",
         _delta_str(kpis['Churn Rate (%)'], base_kpis['Churn Rate (%)'], '%', invert=True)),
        ("Retention Rate", f"{kpis['Retention Rate (%)']:.1f}%",
         _delta_str(kpis['Retention Rate (%)'], base_kpis['Retention Rate (%)'], '%')),
        ("Avg. Products / Customer", f"{kpis['Average Products']:.2f}",
         _delta_str(kpis['Average Products'], base_kpis['Average Products'])),
    ])
    render_kpi_row([
        ("Active Customers", f"{kpis['Active Customer (%)']:.1f}%",
         _delta_str(kpis['Active Customer (%)'], base_kpis['Active Customer (%)'], '%')),
        ("Premium Customers", f"{kpis['Premium Customer (%)']:.1f}%",
         _delta_str(kpis['Premium Customer (%)'], base_kpis['Premium Customer (%)'], '%')),
        ("Avg. Balance", f"${kpis['Average Balance']:,.0f}",
         _delta_str(kpis['Average Balance'], base_kpis['Average Balance'])),
        ("Avg. Credit Score", f"{kpis['Average Credit Score']:.0f}",
         _delta_str(kpis['Average Credit Score'], base_kpis['Average Credit Score'])),
    ])

    render_benchmark_panel()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            filtered_df, names="ChurnLabel", hole=0.45, title="Churn vs Retention",
            color="ChurnLabel", color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    with col2:
        geo_churn = filtered_df.groupby(["Geography", "ChurnLabel"]).size().reset_index(name="Count")
        fig = px.bar(
            geo_churn, x="Geography", y="Count", color="ChurnLabel", barmode="group",
            title="Churn by Geography", color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                     "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
                     "EngagementScore", "RelationshipStrengthIndex", "RetentionScore"]
    numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]
    corr = filtered_df[numeric_cols].astype(float).corr()
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=az.PLOTLY_TEMPLATE, aspect="auto",
    )
    fig.update_layout(height=760)
    fig.update_xaxes(tickfont={"size": 12})
    fig.update_yaxes(tickfont={"size": 12})
    render_plot(fig)

    st.subheader("⚠️ Risk Alerts")
    at_risk_pct = (filtered_df["AtRiskPremiumCustomer"] == "Yes").mean() * 100
    high_bal_disengage = kpis["High Balance Disengagement Rate (%)"]
    if at_risk_pct > 0:
        st.markdown(
            f'<div class="risk-alert">🚨 {at_risk_pct:.1f}% of customers in the current '
            "filter are At-Risk Premium Customers -- high-value, low-engagement accounts "
            "requiring immediate relationship-manager attention.</div>",
            unsafe_allow_html=True,
        )
    if high_bal_disengage > 20:
        st.markdown(
            f'<div class="risk-alert">🚨 {high_bal_disengage:.1f}% of high-balance customers '
            "are currently inactive -- a significant disengagement risk among the bank's "
            "most valuable depositors.</div>",
            unsafe_allow_html=True,
        )

    render_action_queue()

    render_footer()


# ==========================================================================
# PAGE 2: CUSTOMER ANALYSIS
# ==========================================================================

def page_customer_analysis() -> None:
    """Render the customer demographic and search / table exploration page."""
    render_page_header(
        "Customer Analysis",
        "Demographic breakdown, search, and detailed customer records.",
        "Customer View",
    )

    render_kpi_row([
        ("Customers in View", f"{len(filtered_df):,}", None),
        ("Avg. Age", f"{filtered_df['Age'].mean():.1f}", None),
        ("Avg. Tenure", f"{filtered_df['Tenure'].mean():.1f} yrs", None),
        ("Avg. Salary", f"${filtered_df['EstimatedSalary'].mean():,.0f}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="Age", color="ChurnLabel", nbins=30, barmode="overlay",
            opacity=0.7, title="Age Distribution", color_discrete_map=CHURN_COLOR_MAP,
            template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)
    with col2:
        gender_churn = filtered_df.groupby(["Gender", "ChurnLabel"]).size().reset_index(name="Count")
        fig = px.bar(
            gender_churn, x="Gender", y="Count", color="ChurnLabel", barmode="group",
            title="Gender vs Churn", color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    st.subheader("🔍 Search Customer")
    search_term = st.text_input("Search by Customer ID or Surname")
    display_df = filtered_df.copy()
    if search_term:
        mask = (
            display_df["CustomerId"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["Surname"].astype(str).str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]

    st.subheader("Customer Data Table")
    st.dataframe(display_df, use_container_width=True, height=400)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Download Cleaned Data (CSV)",
            data=full_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_customer_data.csv",
            mime="text/csv",
        )
    with col_b:
        kpi_export_df = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
        st.download_button(
            "⬇️ Download KPI Report (CSV)",
            data=kpi_export_df.to_csv(index=False).encode("utf-8"),
            file_name="kpi_report.csv",
            mime="text/csv",
        )

    render_footer()


# ==========================================================================
# PAGE 3: ENGAGEMENT ANALYTICS
# ==========================================================================

def page_engagement_analytics() -> None:
    """Render engagement score and relationship strength analytics."""
    render_page_header(
        "Engagement Analytics",
        "Composite engagement scoring and relationship-depth analysis.",
        "Engagement View",
    )

    render_kpi_row([
        ("Avg. Engagement Score", f"{filtered_df['EngagementScore'].mean():.1f} / 100", None),
        ("Avg. Relationship Strength", f"{filtered_df['RelationshipStrengthIndex'].mean():.1f} / 100", None),
        ("Engagement/Retention Ratio", f"{kpis['Engagement Retention Ratio']}", None),
        ("Active Members", f"{kpis['Active Customer (%)']:.1f}%", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="EngagementScore", color="ChurnLabel", nbins=30, barmode="overlay",
            opacity=0.7, title="Engagement Score Distribution", color_discrete_map=CHURN_COLOR_MAP,
            template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)
    with col2:
        fig = px.histogram(
            filtered_df, x="RelationshipStrengthIndex", nbins=30,
            title="Relationship Strength Index Distribution",
            color_discrete_sequence=[PRIMARY], template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    st.subheader("Engagement by Customer Lifetime Segment")
    seg_engagement = filtered_df.groupby("CustomerLifetimeSegment")["EngagementScore"].mean().reset_index()
    fig = px.bar(
        seg_engagement, x="CustomerLifetimeSegment", y="EngagementScore",
        title="Average Engagement Score by Lifetime Segment",
        color_discrete_sequence=[PRIMARY], template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    st.subheader("Engagement Profile Distribution")
    profile_counts = filtered_df["EngagementProfile"].value_counts().reset_index()
    profile_counts.columns = ["EngagementProfile", "Count"]
    fig = px.bar(
        profile_counts, x="EngagementProfile", y="Count",
        title="Required Engagement Profiles",
        color="EngagementProfile",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    st.subheader("Active Membership vs Churn")
    active_labels = filtered_df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    active_churn = pd.DataFrame({"ActiveLabel": active_labels, "ChurnLabel": filtered_df["ChurnLabel"]})
    active_churn = active_churn.groupby(["ActiveLabel", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        active_churn, x="ActiveLabel", y="Count", color="ChurnLabel", barmode="group",
        color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    render_footer()


# ==========================================================================
# PAGE 4: PRODUCT UTILIZATION
# ==========================================================================

def page_product_utilization() -> None:
    """Render product ownership and utilization analytics."""
    render_page_header(
        "Product Utilization",
        "Product-holding patterns and their relationship to churn.",
        "Product View",
    )

    render_kpi_row([
        ("Avg. Products / Customer", f"{kpis['Average Products']:.2f}", None),
        ("Product Utilization Index", f"{kpis['Product Utilization Index']:.1f}%", None),
        ("Credit Card Stickiness", f"{kpis['Credit Card Stickiness Score (%)']:.1f}%", None),
        ("Avg. Product Depth Index", f"{filtered_df['ProductDepthIndex'].mean():.2f}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        prod_counts = filtered_df["NumOfProducts"].value_counts().sort_index().reset_index()
        prod_counts.columns = ["NumOfProducts", "Count"]
        fig = px.bar(
            prod_counts, x="NumOfProducts", y="Count", title="Product Ownership Distribution",
            color_discrete_sequence=[PRIMARY], template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)
    with col2:
        prod_churn = filtered_df.groupby(["NumOfProducts", "ChurnLabel"]).size().reset_index(name="Count")
        fig = px.bar(
            prod_churn, x="NumOfProducts", y="Count", color="ChurnLabel", barmode="group",
            title="Products vs Churn", color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    st.subheader("Credit Card Ownership vs Churn")
    cc_labels = filtered_df["HasCrCard"].map({1: "Has Credit Card", 0: "No Credit Card"})
    cc_churn = pd.DataFrame({"CrCardLabel": cc_labels, "ChurnLabel": filtered_df["ChurnLabel"]})
    cc_churn = cc_churn.groupby(["CrCardLabel", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        cc_churn, x="CrCardLabel", y="Count", color="ChurnLabel", barmode="group",
        color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    st.subheader("Single-Product vs Multi-Product Retention")
    single_prod = filtered_df[filtered_df["NumOfProducts"] == 1]
    multi_prod = filtered_df[filtered_df["NumOfProducts"] >= 2]

    single_retention = (single_prod["Exited"] == 0).mean() * 100 if len(single_prod) > 0 else 0
    multi_retention = (multi_prod["Exited"] == 0).mean() * 100 if len(multi_prod) > 0 else 0
    single_churn = 100 - single_retention
    multi_churn = 100 - multi_retention

    col_sp, col_mp = st.columns(2)
    with col_sp:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Single-Product Customers (1 Product)</div>
                <div class="metric-value">{single_retention:.1f}%</div>
                <div class="metric-delta" style="color:{DANGER};">Churn: {single_churn:.1f}% · {len(single_prod):,} customers</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_mp:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Multi-Product Customers (2+ Products)</div>
                <div class="metric-value">{multi_retention:.1f}%</div>
                <div class="metric-delta" style="color:{DANGER};">Churn: {multi_churn:.1f}% · {len(multi_prod):,} customers</div>
            </div>""",
            unsafe_allow_html=True,
        )

    retention_comparison = pd.DataFrame({
        "Segment": ["Single-Product (1)", "Multi-Product (2+)"],
        "Retention Rate (%)": [single_retention, multi_retention],
        "Churn Rate (%)": [single_churn, multi_churn],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=retention_comparison["Segment"], y=retention_comparison["Retention Rate (%)"],
        name="Retained", marker_color=SUCCESS,
    ))
    fig.add_trace(go.Bar(
        x=retention_comparison["Segment"], y=retention_comparison["Churn Rate (%)"],
        name="Churned", marker_color=DANGER,
    ))
    fig.update_layout(
        title="Single-Product vs Multi-Product: Retention & Churn Rates",
        barmode="group", template=az.PLOTLY_TEMPLATE,
        yaxis_title="Rate (%)",
    )
    render_plot(fig)

    render_footer()


# ==========================================================================
# PAGE 5: PREMIUM CUSTOMER DETECTOR
# ==========================================================================

def page_premium_customer_detector() -> None:
    """Render premium customer identification and at-risk-premium analysis."""
    render_page_header(
        "Premium Customer Detector",
        "Identify high-value customers and flag those at risk of churn.",
        "Priority View",
    )

    premium_df = filtered_df[filtered_df["PremiumCustomer"] == "Yes"]
    at_risk_df = filtered_df[filtered_df["AtRiskPremiumCustomer"] == "Yes"]

    render_kpi_row([
        ("Premium Customers", f"{len(premium_df):,}", None),
        ("Premium Customer %", f"{kpis['Premium Customer (%)']:.1f}%", None),
        ("At-Risk Premium Customers", f"{len(at_risk_df):,}", None),
        ("At-Risk Premium %", f"{(len(at_risk_df) / len(filtered_df) * 100):.1f}%", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        premium_risk = filtered_df.groupby(["PremiumCustomer", "AtRiskPremiumCustomer"]).size().reset_index(name="Count")
        fig = px.bar(
            premium_risk, x="PremiumCustomer", y="Count", color="AtRiskPremiumCustomer",
            barmode="group", title="Premium Customer Risk Breakdown",
            color_discrete_map={"Yes": DANGER, "No": SUCCESS}, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)
    with col2:
        fig = px.scatter(
            filtered_df, x="EstimatedSalary", y="Balance", color="PremiumCustomer", opacity=0.5,
            title="Premium Customer Map (Salary vs Balance)",
            color_discrete_map={"Yes": ACCENT, "No": NEUTRAL}, template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    if len(at_risk_df) > 0:
        st.markdown(
            f'<div class="risk-alert">🚨 {len(at_risk_df)} high-value customers are '
            "currently inactive or under-engaged. These accounts should be prioritized "
            "for retention outreach given their revenue significance.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("At-Risk Premium Customer Records")
    risk_cols = [
        "CustomerId", "Surname", "Geography", "Balance", "EstimatedSalary",
        "EngagementScore", "IsActiveMember",
    ]
    risk_cols = [c for c in risk_cols if c in at_risk_df.columns]
    st.dataframe(at_risk_df[risk_cols], use_container_width=True, height=350)

    st.subheader("Salary vs Balance Mismatch Detector")
    mismatch_df = filtered_df[filtered_df["SalaryBalanceMismatch"] == "Yes"]
    st.markdown(
        f"<div class='insight-card'>\n"
        f"<strong>Mismatch Rate:</strong> {(len(mismatch_df) / len(filtered_df) * 100):.1f}% of the current view\n"
        f"<br><strong>Definition:</strong> High balance with low salary, or high salary with low balance\n"
        f"</div>",
        unsafe_allow_html=True,
    )
    if len(mismatch_df) > 0:
        mismatch_cols = ["CustomerId", "Surname", "Geography", "Balance", "EstimatedSalary", "SalaryBalanceMismatchType"]
        mismatch_cols = [c for c in mismatch_cols if c in mismatch_df.columns]
        st.dataframe(mismatch_df[mismatch_cols], use_container_width=True, height=220)

    render_footer()


# ==========================================================================
# PAGE 6: RETENTION ANALYTICS
# ==========================================================================

def page_retention_analytics() -> None:
    """Render retention scoring, sticky customer, and risk-level analytics."""
    render_page_header(
        "Retention Analytics",
        "Retention scoring, sticky-customer behavior, and churn risk segmentation.",
        "Retention View",
    )

    render_kpi_row([
        ("Retention Rate", f"{kpis['Retention Rate (%)']:.1f}%", None),
        ("Avg. Retention Score", f"{filtered_df['RetentionScore'].mean():.1f} / 100", None),
        ("Sticky Customers", f"{(filtered_df['RetentionStabilityTier'] == 'Sticky').sum():,}", None),
        ("Stable Customers", f"{(filtered_df['RetentionStabilityTier'] == 'Stable').sum():,}", None),
    ])

    st.info(
        f"Sticky customers meet the explicit threshold of active + credit card + tenure >= {az.STICKY_CUSTOMER_MIN_TENURE} "
        f"and engagement score >= {az.STICKY_CUSTOMER_MIN_ENGAGEMENT}. Stable customers are defined as RetentionScore >= {az.STABLE_CUSTOMER_MIN_RETENTION_SCORE}.",
        icon="ℹ️",
    )

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = filtered_df["RiskLevel"].value_counts().reindex(
            ["Low Risk", "Medium Risk", "High Risk"]
        ).reset_index()
        risk_counts.columns = ["RiskLevel", "Count"]
        fig = px.bar(
            risk_counts, x="RiskLevel", y="Count", title="Customer Risk Level Distribution",
            color="RiskLevel",
            color_discrete_map={"Low Risk": SUCCESS, "Medium Risk": WARNING, "High Risk": DANGER},
            template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)
    with col2:
        sticky_churn = filtered_df.groupby(["StickyCustomer", "ChurnLabel"]).size().reset_index(name="Count")
        fig = px.bar(
            sticky_churn, x="StickyCustomer", y="Count", color="ChurnLabel", barmode="group",
            title="Sticky Customer Analysis", color_discrete_map=CHURN_COLOR_MAP,
            template=az.PLOTLY_TEMPLATE,
        )
        render_plot(fig)

    st.subheader("Behavioral Profiles")
    profile_counts = filtered_df["EngagementProfile"].value_counts().reset_index()
    profile_counts.columns = ["EngagementProfile", "Count"]
    fig = px.bar(
        profile_counts, x="EngagementProfile", y="Count", color="EngagementProfile",
        color_discrete_sequence=px.colors.qualitative.Set2, template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    st.subheader("Balance Distribution by Churn Status")
    fig = px.box(
        filtered_df, x="ChurnLabel", y="Balance", color="ChurnLabel",
        color_discrete_map=CHURN_COLOR_MAP, template=az.PLOTLY_TEMPLATE,
    )
    render_plot(fig)

    st.subheader("📋 Recommendations")
    recommendations = [
        "Launch a targeted reactivation campaign for inactive members, "
        "who churn at roughly double the rate of active members.",
        "Prioritize relationship-manager outreach to At-Risk Premium "
        "Customers before they show explicit churn signals.",
        "Investigate market-specific drivers in the highest-churn "
        "geography identified in Business Insights.",
        "Avoid over-selling products beyond 2 per customer without a "
        "clear needs-based rationale, since 3-4 product holders show "
        "sharply elevated churn.",
    ]
    for rec in recommendations:
        st.markdown(f'<div class="recommendation-card">✅ {rec}</div>', unsafe_allow_html=True)

    render_action_queue()

    render_footer()


# ==========================================================================
# PAGE 7: BUSINESS INSIGHTS
# ==========================================================================

def page_business_insights() -> None:
    """Render the automatically generated business insight summary page."""
    render_page_header(
        "Business Insights",
        "Automatically generated, data-driven insights for the current filter selection.",
        "Strategic View",
    )
    st.caption(f"🕐 Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Top KPI summary for quick executive view
    sample_n = len(filtered_df)
    exec_kpis = [
        ("Customers in View", f"{kpis.get('Total Customers', sample_n):,}", None),
        ("Churn Rate", f"{kpis.get('Churn Rate (%)', 0):.1f}%", None),
        ("Retention Rate", f"{kpis.get('Retention Rate (%)', 0):.1f}%", None),
        ("Premium %", f"{kpis.get('Premium Customer (%)', 0):.1f}%", None),
    ]
    render_kpi_row(exec_kpis)

    st.markdown("---")
    st.subheader("Top Insights")

    # Simple, stable confidence heuristic based on sample size
    if sample_n > 500:
        confidence_tag = "High"
    elif sample_n > 100:
        confidence_tag = "Medium"
    else:
        confidence_tag = "Low"

    # Show top 5 insights with expanders and a small metadata footer
    top_insights = insights[:5] if insights else ["No automated insights available for this selection."]
    for i, insight in enumerate(top_insights, start=1):
        with st.expander(f"{i}. {insight.split('.')[0][:120]}...", expanded=(i == 1)):
            st.write(insight)
            st.markdown(
                f"**Evidence:** Derived from filtered dataset of {sample_n} customers — **Confidence:** {confidence_tag}"
            )

    st.markdown("---")
    st.subheader("Executive Summary & Export")

    # Build a concise executive summary automatically
    exec_lines = [
        f"Customers in view: {sample_n}",
        f"Churn Rate: {kpis.get('Churn Rate (%)', 0):.1f}%",
        f"Retention Rate: {kpis.get('Retention Rate (%)', 0):.1f}%",
    ]
    exec_summary = "\n".join(exec_lines) + "\n\nTop insights:\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(top_insights, start=1)])

    st.text_area("Edit Executive Summary", value=exec_summary, height=220, key="exec_summary_text")

    st.download_button(
        "⬇️ Download Executive Summary (TXT)",
        data=st.session_state.exec_summary_text.encode("utf-8"),
        file_name="executive_summary.txt",
        mime="text/plain",
    )

    # Also allow CSV download of insights
    st.download_button(
        "⬇️ Download Insights (CSV)",
        data=pd.DataFrame({"Insight #": range(1, len(insights) + 1), "Business Insight": insights}).to_csv(index=False).encode("utf-8"),
        file_name="business_insights.csv",
        mime="text/csv",
    )

    render_footer()


# ==========================================================================
# ROUTER
# ==========================================================================

PAGE_ROUTER = {
    "📊 Executive Dashboard": page_executive_dashboard,
    "👥 Customer Analysis": page_customer_analysis,
    "💡 Engagement Analytics": page_engagement_analytics,
    "📦 Product Utilization": page_product_utilization,
    "💎 Premium Customer Detector": page_premium_customer_detector,
    "🔄 Retention Analytics": page_retention_analytics,
    "📈 Business Insights": page_business_insights,
}

# Render pages inside a polished top navigation using Streamlit tabs.
try:
    tab_objs = st.tabs(NAV_OPTIONS)
    for label, tab in zip(NAV_OPTIONS, tab_objs):
        with tab:
            PAGE_ROUTER[label]()
except Exception:
    # Fallback router if tabs are not available in the deployed Streamlit version.
    selected_page = st.sidebar.radio("Navigation", NAV_OPTIONS, index=0)
    PAGE_ROUTER.get(selected_page, page_executive_dashboard)()
