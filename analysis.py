from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Headless backend -> safe for script / server execution
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Suppress only specific known warnings that don't indicate real issues
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Kaleido.*")

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

__all__ = [
    "load_data",
    "validate_data",
    "clean_data",
    "engineer_features",
    "calculate_kpis",
    "generate_all_charts",
    "generate_business_insights",
    "export_all",
    "run_full_pipeline",
    "COLOR_PALETTE",
    "CHURN_COLOR_MAP",
    "PLOTLY_TEMPLATE",
]

# ------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
RAW_CSV_CANDIDATES = ["European_Bank.csv", "Churn_Modelling.csv", "data.csv"]

# Explicit thresholds used by the retention methodology.
ENGAGEMENT_TIER_BINS = [-0.01, 39.99, 69.99, 100.01]
ENGAGEMENT_TIER_LABELS = ["Low", "Medium", "High"]
STICKY_CUSTOMER_MIN_TENURE = 5
STICKY_CUSTOMER_MIN_ENGAGEMENT = 60
STABLE_CUSTOMER_MIN_RETENTION_SCORE = 55
STRONG_RETENTION_SCORE = 70

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Premium Enterprise SaaS Color Palette (Apple/Stripe/Fabric style)
COLOR_PALETTE = {
    "primary": "#2563EB",       # exact primary requested
    "secondary": "#475569",     # slate grey
    "accent": "#2563EB",        # professional blue accent
    "success": "#10B981",       # exact success requested
    "danger": "#EF4444",        # exact danger requested
    "warning": "#F59E0B",       # exact warning requested
    "neutral": "#E2E8F0",       # soft light grey
    "background": "#F8FAFC",    # exact background requested
}

CHURN_COLOR_MAP = {"Retained": COLOR_PALETTE["success"], "Churned": COLOR_PALETTE["danger"]}
PLOTLY_TEMPLATE = "plotly_white"
CHART_WIDTH = 950
CHART_HEIGHT = 600

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "none",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#F1F5F9",
    }
)


def _log(section: str, message: str) -> None:
    """Print a professionally formatted, timestamped-style log line.

    Parameters
    ----------
    section : str
        Short tag identifying the pipeline stage (e.g. 'VALIDATION').
    message : str
        Human readable log message.
    """
    print(f"[{section:^18}] {message}")


def _log_header(title: str) -> None:
    """Print a section header banner to visually separate pipeline stages."""
    bar = "=" * 78
    print(f"\n{bar}\n {title}\n{bar}")


# ==========================================================================
# 1. DATA LOADING & COLUMN DETECTION
# ==========================================================================

# Canonical column names -> list of acceptable aliases (case-insensitive,
# whitespace-insensitive). This makes the pipeline resilient to minor naming
# variations across different exports of the same underlying dataset.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "CustomerId": ["customerid", "customer_id", "id", "cust_id"],
    "Surname": ["surname", "lastname", "last_name"],
    "CreditScore": ["creditscore", "credit_score"],
    "Geography": ["geography", "country", "region"],
    "Gender": ["gender", "sex"],
    "Age": ["age"],
    "Tenure": ["tenure", "years_with_bank"],
    "Balance": ["balance", "account_balance"],
    "NumOfProducts": ["numofproducts", "num_of_products", "products", "n_products"],
    "HasCrCard": ["hascrcard", "has_cr_card", "has_credit_card", "creditcard"],
    "IsActiveMember": ["isactivemember", "is_active_member", "active", "is_active"],
    "EstimatedSalary": ["estimatedsalary", "estimated_salary", "salary"],
    "Exited": ["exited", "churn", "churned", "is_churned"],
}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Automatically map canonical column names to the actual dataframe columns.

    This allows the pipeline to work even if the raw CSV uses slightly
    different naming conventions (e.g. 'Churn' instead of 'Exited').

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    Dict[str, str]
        Mapping of canonical_name -> actual_column_name_in_df.
    """
    normalized = {c.lower().replace(" ", "").replace("-", "_"): c for c in df.columns}
    mapping: Dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = alias.lower().replace(" ", "").replace("-", "_")
            if key in normalized:
                found = normalized[key]
                break
        if found is not None:
            mapping[canonical] = found

    missing = [c for c in COLUMN_ALIASES if c not in mapping]
    if missing:
        _log("COLUMN MAP", f"WARNING - could not auto-detect columns: {missing}")
    else:
        _log("COLUMN MAP", "All expected columns detected successfully.")

    return mapping


# Minimum columns required for the pipeline to produce meaningful output
MINIMUM_REQUIRED_COLUMNS = ["CreditScore", "Age", "Tenure", "Balance",
                            "NumOfProducts", "IsActiveMember", "Exited"]


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw customer dataset from disk and standardize column names.

    Parameters
    ----------
    path : Optional[str]
        Explicit path to the CSV file. If None, the function searches the
        base directory for a set of known candidate filenames.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with canonical column names applied.

    Raises
    ------
    FileNotFoundError
        If no suitable CSV file is found.
    ValueError
        If the loaded CSV is missing critical columns required by the pipeline.
    """
    _log_header("STAGE 1: DATA LOADING")

    if path is None:
        for candidate in RAW_CSV_CANDIDATES:
            candidate_path = os.path.join(BASE_DIR, candidate)
            if os.path.exists(candidate_path):
                path = candidate_path
                break

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "No source CSV found. Place the dataset in the project folder "
            f"as one of: {RAW_CSV_CANDIDATES}"
        )

    df = pd.read_csv(path)
    _log("LOAD", f"Loaded raw dataset from '{os.path.basename(path)}' -> shape={df.shape}")

    mapping = detect_columns(df)
    inverse_mapping = {v: k for k, v in mapping.items()}
    df = df.rename(columns=inverse_mapping)

    # Validate that all critical columns are present after mapping
    missing_critical = [c for c in MINIMUM_REQUIRED_COLUMNS if c not in df.columns]
    if missing_critical:
        raise ValueError(
            f"Dataset is missing critical columns after auto-detection: {missing_critical}. "
            "The pipeline requires at least: " + ", ".join(MINIMUM_REQUIRED_COLUMNS)
        )
    _log("VALIDATION", "All minimum required columns present after column mapping.")

    return df


# ==========================================================================
# 2. DATA VALIDATION
# ==========================================================================

def validate_data(df: pd.DataFrame) -> Dict[str, object]:
    """Run a full data-quality audit and print professional diagnostic logs.

    Checks performed
    -----------------
    - Missing values per column
    - Duplicate rows / duplicate CustomerIds
    - Invalid binary columns (HasCrCard, IsActiveMember, Exited not in {0,1})
    - Wrong datatypes
    - Impossible ages (<18 or >100)
    - Negative balances
    - Negative salaries
    - Invalid product counts (<1 or >4)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.

    Returns
    -------
    Dict[str, object]
        Summary dictionary of all validation findings (used later for the
        business summary export).
    """
    _log_header("STAGE 2: DATA VALIDATION")

    report: Dict[str, object] = {}

    # --- Missing values -----------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report["missing_values"] = missing.to_dict()
    if missing.empty:
        _log("MISSING", "No missing values detected across any column.")
    else:
        for col, count in missing.items():
            _log("MISSING", f"Column '{col}' has {count} missing values ({count/len(df)*100:.2f}%).")

    # --- Duplicates ------------------------------------------------------
    dup_rows = df.duplicated().sum()
    report["duplicate_rows"] = int(dup_rows)
    _log("DUPLICATES", f"Found {dup_rows} fully duplicated rows.")

    if "CustomerId" in df.columns:
        dup_ids = df["CustomerId"].duplicated().sum()
        report["duplicate_customer_ids"] = int(dup_ids)
        _log("DUPLICATES", f"Found {dup_ids} duplicate CustomerId values.")

    # --- Invalid binary columns -----------------------------------------
    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        if col in df.columns:
            invalid = df[~df[col].isin([0, 1])].shape[0]
            report[f"invalid_binary_{col}"] = int(invalid)
            if invalid > 0:
                _log("BINARY CHECK", f"Column '{col}' has {invalid} values outside {{0,1}}.")
            else:
                _log("BINARY CHECK", f"Column '{col}' is valid binary (0/1 only).")

    # --- Datatype audit --------------------------------------------------
    expected_numeric = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
    ]
    wrong_dtype_cols = []
    for col in expected_numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            wrong_dtype_cols.append(col)
    report["wrong_dtype_columns"] = wrong_dtype_cols
    if wrong_dtype_cols:
        _log("DTYPE", f"Columns with unexpected non-numeric dtype: {wrong_dtype_cols}")
    else:
        _log("DTYPE", "All expected numeric columns have correct datatypes.")

    # --- Impossible ages ---------------------------------------------------
    if "Age" in df.columns:
        impossible_age = df[(df["Age"] < 18) | (df["Age"] > 100)].shape[0]
        report["impossible_ages"] = int(impossible_age)
        _log("AGE CHECK", f"Found {impossible_age} records with impossible ages (<18 or >100).")

    # --- Negative balances ---------------------------------------------------
    if "Balance" in df.columns:
        neg_balance = df[df["Balance"] < 0].shape[0]
        report["negative_balances"] = int(neg_balance)
        _log("BALANCE CHECK", f"Found {neg_balance} records with negative balances.")

    # --- Negative salaries ---------------------------------------------------
    if "EstimatedSalary" in df.columns:
        neg_salary = df[df["EstimatedSalary"] < 0].shape[0]
        report["negative_salaries"] = int(neg_salary)
        _log("SALARY CHECK", f"Found {neg_salary} records with negative salaries.")

    # --- Invalid product counts -----------------------------------------
    if "NumOfProducts" in df.columns:
        invalid_products = df[(df["NumOfProducts"] < 1) | (df["NumOfProducts"] > 4)].shape[0]
        report["invalid_product_counts"] = int(invalid_products)
        _log("PRODUCT CHECK", f"Found {invalid_products} records with invalid product counts.")

    _log("VALIDATION", "Validation pass complete.")
    return report


# ==========================================================================
# 3. DATA CLEANING
# ==========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe based on validation findings.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Drop duplicate CustomerId rows (keep first occurrence).
    3. Impute missing numeric values with column median; categorical with mode.
    4. Coerce binary / numeric columns to correct datatypes.
    5. Clip impossible ages to the [18, 100] range.
    6. Clip negative balances / salaries to zero.
    7. Clip invalid product counts to the valid [1, 4] range.
    8. Treat extreme outliers (using IQR) for Balance & EstimatedSalary by
       capping rather than deleting, to preserve sample size for a banking
       dataset where legitimate high-net-worth customers are expected.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (post-validation).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    _log_header("STAGE 3: DATA CLEANING")

    clean_df = df.copy()
    initial_rows = len(clean_df)

    # --- Remove duplicates -------------------------------------------------
    clean_df.drop_duplicates(inplace=True)
    if "CustomerId" in clean_df.columns:
        clean_df.drop_duplicates(subset="CustomerId", keep="first", inplace=True)
    removed = initial_rows - len(clean_df)
    _log("DEDUP", f"Removed {removed} duplicate rows -> {len(clean_df)} rows remain.")

    # --- Handle missing values ----------------------------------------------
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    categorical_cols = clean_df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if clean_df[col].isnull().sum() > 0:
            median_val = clean_df[col].median()
            clean_df[col].fillna(median_val, inplace=True)
            _log("IMPUTE", f"Filled missing numeric values in '{col}' with median={median_val}.")

    for col in categorical_cols:
        if clean_df[col].isnull().sum() > 0:
            mode_val = clean_df[col].mode().iloc[0]
            clean_df[col].fillna(mode_val, inplace=True)
            _log("IMPUTE", f"Filled missing categorical values in '{col}' with mode='{mode_val}'.")

    # --- Datatype correction -------------------------------------------------
    int_cols = ["CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]
    for col in int_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").round().astype("Int64")

    float_cols = ["Balance", "EstimatedSalary"]
    for col in float_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").astype(float)

    # --- Fix impossible ages -------------------------------------------------
    if "Age" in clean_df.columns:
        before = clean_df[(clean_df["Age"] < 18) | (clean_df["Age"] > 100)].shape[0]
        clean_df["Age"] = clean_df["Age"].clip(lower=18, upper=100)
        _log("AGE FIX", f"Clipped {before} impossible age values into the [18, 100] range.")

    # --- Fix negative balances / salaries -------------------------------------
    if "Balance" in clean_df.columns:
        before = clean_df[clean_df["Balance"] < 0].shape[0]
        clean_df["Balance"] = clean_df["Balance"].clip(lower=0)
        _log("BALANCE FIX", f"Clipped {before} negative balances to 0.")

    if "EstimatedSalary" in clean_df.columns:
        before = clean_df[clean_df["EstimatedSalary"] < 0].shape[0]
        clean_df["EstimatedSalary"] = clean_df["EstimatedSalary"].clip(lower=0)
        _log("SALARY FIX", f"Clipped {before} negative salaries to 0.")

    # --- Fix invalid product counts -----------------------------------------
    if "NumOfProducts" in clean_df.columns:
        before = clean_df[(clean_df["NumOfProducts"] < 1) | (clean_df["NumOfProducts"] > 4)].shape[0]
        clean_df["NumOfProducts"] = clean_df["NumOfProducts"].clip(lower=1, upper=4)
        _log("PRODUCT FIX", f"Clipped {before} invalid product counts into the [1, 4] range.")

    # --- Outlier treatment (IQR capping) for Balance & Salary -----------------
    for col in ["Balance", "EstimatedSalary", "CreditScore"]:
        if col in clean_df.columns:
            q1, q3 = clean_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = clean_df[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)].shape[0]
            clean_df[col] = clean_df[col].clip(lower=max(lower_bound, 0), upper=upper_bound)
            _log("OUTLIER CAP", f"Capped {outliers} outliers in '{col}' using IQR bounds.")

    clean_df.reset_index(drop=True, inplace=True)
    _log("CLEANING", f"Cleaning complete. Final shape={clean_df.shape}.")

    return clean_df


# ==========================================================================
# 4. FEATURE ENGINEERING
# ==========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived business features used throughout KPI and dashboard logic.

    New features
    ------------
    - Engagement Score          : weighted composite of activity, tenure, products, credit card
    - Relationship Strength     : composite of tenure, products, balance-to-salary ratio
    - Product Depth Index       : normalized NumOfProducts (0-1 scale)
    - Premium Customer          : Balance & EstimatedSalary above 75th percentile
    - Sticky Customer           : Active + Credit Card + Tenure >= 5 years
    - At Risk Premium Customer  : Premium AND (inactive OR low engagement)
    - Customer Lifetime Segment : New / Growing / Established / Veteran (by tenure)
    - Age Group                 : Young Adult / Adult / Middle Aged / Senior
    - Balance Category          : Zero / Low / Medium / High
    - Salary Category           : Low / Medium / High
    - Risk Level                : Low / Medium / High churn risk (heuristic score)
    - Retention Score           : 0-100 composite likelihood-to-stay score

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with all engineered features.
    """
    _log_header("STAGE 4: FEATURE ENGINEERING")

    feat_df = df.copy()

    # --- Engagement Score (0-100) ------------------------------------------
    # Weighted blend of activity status, credit card ownership, product count
    # and tenure -> a single number describing how "engaged" a customer is.
    norm_tenure = feat_df["Tenure"] / feat_df["Tenure"].max()
    norm_products = (feat_df["NumOfProducts"] - 1) / 3  # scale 1-4 -> 0-1
    feat_df["EngagementScore"] = (
        feat_df["IsActiveMember"].astype(float) * 40
        + feat_df["HasCrCard"].astype(float) * 15
        + norm_products * 25
        + norm_tenure * 20
    ).round(2)
    _log("FEATURE", "Created 'EngagementScore' (0-100 composite).")

    # --- Engagement Tier ----------------------------------------------------
    feat_df["EngagementTier"] = pd.cut(
        feat_df["EngagementScore"],
        bins=ENGAGEMENT_TIER_BINS,
        labels=ENGAGEMENT_TIER_LABELS,
        include_lowest=True,
    )
    _log("FEATURE", "Created 'EngagementTier' (Low/Medium/High engagement bands).")

    # --- Relationship Strength Index (0-100) --------------------------------
    balance_to_salary = (feat_df["Balance"] / feat_df["EstimatedSalary"].replace(0, np.nan)).fillna(0)
    balance_to_salary_capped = balance_to_salary.clip(upper=balance_to_salary.quantile(0.95))
    norm_bts = balance_to_salary_capped / balance_to_salary_capped.max() if balance_to_salary_capped.max() > 0 else 0
    feat_df["RelationshipStrengthIndex"] = (
        norm_tenure * 35 + norm_products * 35 + norm_bts * 30
    ).round(2)
    _log("FEATURE", "Created 'RelationshipStrengthIndex' (0-100 composite).")

    # --- Product Depth Index -------------------------------------------------
    feat_df["ProductDepthIndex"] = norm_products.round(2)
    _log("FEATURE", "Created 'ProductDepthIndex' (normalized 0-1).")

    # --- Premium Customer -----------------------------------------------------
    balance_75 = feat_df["Balance"].quantile(0.75)
    balance_25 = feat_df["Balance"].quantile(0.25)
    salary_75 = feat_df["EstimatedSalary"].quantile(0.75)
    salary_25 = feat_df["EstimatedSalary"].quantile(0.25)
    feat_df["PremiumCustomer"] = np.where(
        (feat_df["Balance"] >= balance_75) & (feat_df["EstimatedSalary"] >= salary_75), "Yes", "No"
    )
    _log("FEATURE", f"Created 'PremiumCustomer' flag (Balance>={balance_75:.0f} & Salary>={salary_75:.0f}).")

    # --- Salary / Balance Mismatch Detector ---------------------------------
    high_balance_low_salary = (feat_df["Balance"] >= balance_75) & (feat_df["EstimatedSalary"] <= salary_25)
    low_balance_high_salary = (feat_df["Balance"] <= balance_25) & (feat_df["EstimatedSalary"] >= salary_75)
    feat_df["SalaryBalanceMismatchType"] = np.select(
        [high_balance_low_salary, low_balance_high_salary],
        ["High Balance / Low Salary", "Low Balance / High Salary"],
        default="None",
    )
    feat_df["SalaryBalanceMismatch"] = np.where(feat_df["SalaryBalanceMismatchType"] != "None", "Yes", "No")
    _log("FEATURE", "Created 'SalaryBalanceMismatch' detector for balance/salary outliers.")

    # --- Sticky Customer -------------------------------------------------------
    feat_df["StickyCustomer"] = np.where(
        (feat_df["IsActiveMember"] == 1)
        & (feat_df["HasCrCard"] == 1)
        & (feat_df["Tenure"] >= STICKY_CUSTOMER_MIN_TENURE)
        & (feat_df["EngagementScore"] >= STICKY_CUSTOMER_MIN_ENGAGEMENT),
        "Yes", "No"
    )
    _log(
        "FEATURE",
        f"Created 'StickyCustomer' flag (Active + Credit Card + Tenure>={STICKY_CUSTOMER_MIN_TENURE} + "
        f"EngagementScore>={STICKY_CUSTOMER_MIN_ENGAGEMENT}).",
    )

    # --- At Risk Premium Customer ------------------------------------------
    feat_df["AtRiskPremiumCustomer"] = np.where(
        (feat_df["PremiumCustomer"] == "Yes")
        & ((feat_df["IsActiveMember"] == 0) | (feat_df["EngagementScore"] < 50)),
        "Yes", "No"
    )
    _log("FEATURE", "Created 'AtRiskPremiumCustomer' flag (Premium + Inactive/Low Engagement).")

    # --- Customer Lifetime Segment ------------------------------------------
    def _lifetime_segment(tenure: float) -> str:
        if tenure <= 2:
            return "New"
        elif tenure <= 5:
            return "Growing"
        elif tenure <= 8:
            return "Established"
        else:
            return "Veteran"

    feat_df["CustomerLifetimeSegment"] = feat_df["Tenure"].apply(_lifetime_segment)
    _log("FEATURE", "Created 'CustomerLifetimeSegment' (New/Growing/Established/Veteran).")

    # --- Age Group ---------------------------------------------------------
    feat_df["AgeGroup"] = pd.cut(
        feat_df["Age"],
        bins=[17, 30, 45, 60, 101],
        labels=["Young Adult (18-30)", "Adult (31-45)", "Middle Aged (46-60)", "Senior (60+)"],
    )
    _log("FEATURE", "Created 'AgeGroup' bucketed categories.")

    # --- Balance Category ----------------------------------------------------
    def _balance_category(bal: float) -> str:
        if bal == 0:
            return "Zero Balance"
        elif bal < feat_df["Balance"].quantile(0.33):
            return "Low Balance"
        elif bal < feat_df["Balance"].quantile(0.66):
            return "Medium Balance"
        else:
            return "High Balance"

    feat_df["BalanceCategory"] = feat_df["Balance"].apply(_balance_category)
    _log("FEATURE", "Created 'BalanceCategory' (Zero/Low/Medium/High).")

    # --- Engagement Profiles -------------------------------------------------
    feat_df["EngagementProfile"] = np.select(
        [
            (feat_df["IsActiveMember"] == 1) & (feat_df["EngagementScore"] >= 65),
            (feat_df["IsActiveMember"] == 1) & (feat_df["NumOfProducts"] == 1),
            (feat_df["IsActiveMember"] == 0) & (feat_df["BalanceCategory"] == "High Balance"),
            (feat_df["IsActiveMember"] == 0) & (feat_df["EngagementScore"] < 50),
        ],
        [
            "Active Engaged",
            "Active but Low-Product",
            "Inactive High-Balance",
            "Inactive Disengaged",
        ],
        default="Other",
    )
    _log("FEATURE", "Created 'EngagementProfile' for the four required behavioral segments.")

    # --- Salary Category -------------------------------------------------------
    try:
        feat_df["SalaryCategory"] = pd.qcut(
            feat_df["EstimatedSalary"], q=3, labels=["Low Salary", "Medium Salary", "High Salary"],
            duplicates="drop",
        )
    except ValueError:
        # Fallback to equal-width bins if quantile edges are not unique
        feat_df["SalaryCategory"] = pd.cut(
            feat_df["EstimatedSalary"], bins=3, labels=["Low Salary", "Medium Salary", "High Salary"],
        )
    _log("FEATURE", "Created 'SalaryCategory' (Low/Medium/High tertiles).")

    # --- Risk Level (heuristic churn risk, not a trained model) --------------
    risk_score = (
        (feat_df["IsActiveMember"] == 0).astype(int) * 3
        + (feat_df["NumOfProducts"] == 1).astype(int) * 2
        + (feat_df["Age"] > 50).astype(int) * 2
        + (feat_df["HasCrCard"] == 0).astype(int) * 1
        + (feat_df["Balance"] == 0).astype(int) * 1
    )
    feat_df["RiskLevel"] = pd.cut(
        risk_score, bins=[-1, 2, 5, 9], labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    _log("FEATURE", "Created 'RiskLevel' heuristic churn-risk classification.")

    # --- Retention Score (0-100, higher = more likely to stay) ---------------
    feat_df["RetentionScore"] = (
        feat_df["EngagementScore"] * 0.5 + feat_df["RelationshipStrengthIndex"] * 0.5
    ).round(2)
    _log("FEATURE", "Created 'RetentionScore' (0-100 composite likelihood-to-stay).")

    # --- Retention Stability Tier -------------------------------------------
    feat_df["RetentionStabilityTier"] = np.select(
        [
            (feat_df["StickyCustomer"] == "Yes") & (feat_df["RetentionScore"] >= STRONG_RETENTION_SCORE),
            feat_df["RetentionScore"] >= STABLE_CUSTOMER_MIN_RETENTION_SCORE,
        ],
        ["Sticky", "Stable"],
        default="Watchlist",
    )
    _log(
        "FEATURE",
        "Created 'RetentionStabilityTier' (Sticky >= "
        f"{STRONG_RETENTION_SCORE}, Stable >= {STABLE_CUSTOMER_MIN_RETENTION_SCORE}).",
    )

    _log("FEATURE ENGINEERING", f"Completed. Total columns now = {feat_df.shape[1]}.")

    return feat_df


# ==========================================================================
# 5. KPI CALCULATIONS
# ==========================================================================

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute the full suite of executive-level KPIs for the dashboard.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe.

    Returns
    -------
    Dict[str, float]
        Dictionary of KPI name -> value.
    """
    _log_header("STAGE 5: KPI CALCULATION")

    total_customers = len(df)
    total_churn = int(df["Exited"].sum())
    churn_rate = round(total_churn / total_customers * 100, 2)
    retention_rate = round(100 - churn_rate, 2)

    active_pct = round((df["IsActiveMember"] == 1).mean() * 100, 2)
    inactive_pct = round(100 - active_pct, 2)
    premium_pct = round((df["PremiumCustomer"] == "Yes").mean() * 100, 2)

    # Product Utilization Index: average products owned relative to max (4)
    max_products = max(int(df["NumOfProducts"].max()), 4)
    product_utilization_index = round((df["NumOfProducts"].mean() / max_products) * 100, 2)

    # Engagement Retention Ratio: Active vs inactive churn comparison
    # Shows how many times more likely inactive customers are to churn vs active ones.
    active_churn_rate = df.loc[df["IsActiveMember"] == 1, "Exited"].mean() * 100
    inactive_churn_rate = df.loc[df["IsActiveMember"] == 0, "Exited"].mean() * 100
    engagement_retention_ratio = (
        round(inactive_churn_rate / active_churn_rate, 2)
        if active_churn_rate > 0
        else np.nan
    )

    # High Balance Disengagement Rate: % of high-balance customers who are inactive
    if "BalanceCategory" in df.columns:
        high_balance_customers = df[df["BalanceCategory"] == "High Balance"]
    else:
        # Fallback: use top-quartile balance as high-balance proxy
        high_balance_customers = df[df["Balance"] >= df["Balance"].quantile(0.75)]
    high_balance_disengagement_rate = (
        round((high_balance_customers["IsActiveMember"] == 0).mean() * 100, 2)
        if len(high_balance_customers) > 0 else 0.0
    )

    # Credit Card Stickiness Score: retention rate among credit card holders
    cc_holders = df[df["HasCrCard"] == 1]
    credit_card_stickiness_score = (
        round((cc_holders["Exited"] == 0).mean() * 100, 2) if len(cc_holders) > 0 else 0.0
    )

    kpis: Dict[str, float] = {
        "Total Customers": total_customers,
        "Total Churn": total_churn,
        "Churn Rate (%)": churn_rate,
        "Retention Rate (%)": retention_rate,
        "Average Balance": round(df["Balance"].mean(), 2),
        "Average Salary": round(df["EstimatedSalary"].mean(), 2),
        "Average Products": round(df["NumOfProducts"].mean(), 2),
        "Average Credit Score": round(df["CreditScore"].mean(), 2),
        "Active Customer (%)": active_pct,
        "Inactive Customer (%)": inactive_pct,
        "Premium Customer (%)": premium_pct,
        "Product Utilization Index": product_utilization_index,
        "Relationship Strength Index": round(df["RelationshipStrengthIndex"].mean(), 2),
        "Engagement Retention Ratio": engagement_retention_ratio,
        "High Balance Disengagement Rate (%)": high_balance_disengagement_rate,
        "Credit Card Stickiness Score (%)": credit_card_stickiness_score,
        "Average Tenure (yrs)": round(df["Tenure"].mean(), 2),
    }

    for name, value in kpis.items():
        _log("KPI", f"{name}: {value}")

    return kpis


# ==========================================================================
# 6. VISUALIZATIONS
# ==========================================================================

def _save_matplotlib(fig: plt.Figure, filename: str) -> None:
    """Save a Matplotlib figure to the charts directory and close it."""
    path = os.path.join(CHARTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _log("CHART SAVED", filename)


def _save_plotly(fig: go.Figure, filename: str) -> None:
    """Save a Plotly figure as a static PNG (requires kaleido) to charts dir."""
    path = os.path.join(CHARTS_DIR, filename)
    try:
        fig.write_image(path, width=CHART_WIDTH, height=CHART_HEIGHT, scale=2)
        _log("CHART SAVED", filename)
    except Exception as exc:  # pragma: no cover - kaleido may be unavailable
        _log("CHART WARNING", f"Could not export '{filename}' as PNG ({exc}). Skipping static export.")


def _map_churn_labels(series: pd.Series) -> pd.Series:
    """Map binary Exited column to readable labels for chart legends."""
    return series.map({0: "Retained", 1: "Churned"})


def generate_all_charts(df: pd.DataFrame) -> Dict[str, object]:
    """Generate the full suite of 20 publication-quality charts.

    Every chart is saved as a PNG in `charts/`. Plotly figures are also
    returned in a dictionary so the Streamlit app can render them
    interactively without regenerating them.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe.

    Returns
    -------
    Dict[str, object]
        Mapping of chart_name -> Plotly figure object (for interactive reuse).
    """
    _log_header("STAGE 6: VISUALIZATION GENERATION")

    figures: Dict[str, object] = {}
    plot_df = df.copy()
    plot_df["ChurnLabel"] = _map_churn_labels(plot_df["Exited"])

    # 1. Churn Distribution ---------------------------------------------------
    fig = px.pie(
        plot_df, names="ChurnLabel", title="Customer Churn Distribution",
        color="ChurnLabel", color_discrete_map=CHURN_COLOR_MAP, hole=0.45,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textinfo="percent+label")
    _save_plotly(fig, "01_churn_distribution.png")
    figures["churn_distribution"] = fig

    # 2. Geography vs Churn ----------------------------------------------------
    geo_churn = plot_df.groupby(["Geography", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        geo_churn, x="Geography", y="Count", color="ChurnLabel", barmode="group",
        title="Churn Distribution by Geography", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "02_geography_vs_churn.png")
    figures["geography_vs_churn"] = fig

    # 3. Gender vs Churn --------------------------------------------------------
    gender_churn = plot_df.groupby(["Gender", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        gender_churn, x="Gender", y="Count", color="ChurnLabel", barmode="group",
        title="Churn Distribution by Gender", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "03_gender_vs_churn.png")
    figures["gender_vs_churn"] = fig

    # 4. Age Distribution ---------------------------------------------------------
    fig = px.histogram(
        plot_df, x="Age", color="ChurnLabel", nbins=30, barmode="overlay", opacity=0.7,
        title="Customer Age Distribution", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "04_age_distribution.png")
    figures["age_distribution"] = fig

    # 5. Balance Distribution ------------------------------------------------------
    fig = px.histogram(
        plot_df, x="Balance", nbins=40, title="Account Balance Distribution",
        color_discrete_sequence=[COLOR_PALETTE["primary"]], template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "05_balance_distribution.png")
    figures["balance_distribution"] = fig

    # 6. Salary Distribution -----------------------------------------------------
    fig = px.histogram(
        plot_df, x="EstimatedSalary", nbins=40, title="Estimated Salary Distribution",
        color_discrete_sequence=[COLOR_PALETTE["accent"]], template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "06_salary_distribution.png")
    figures["salary_distribution"] = fig

    # 7. Products vs Churn -----------------------------------------------------
    prod_churn = plot_df.groupby(["NumOfProducts", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        prod_churn, x="NumOfProducts", y="Count", color="ChurnLabel", barmode="group",
        title="Number of Products vs Churn", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "07_products_vs_churn.png")
    figures["products_vs_churn"] = fig

    # 8. Active Member vs Churn --------------------------------------------------
    plot_df["ActiveLabel"] = plot_df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    active_churn = plot_df.groupby(["ActiveLabel", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        active_churn, x="ActiveLabel", y="Count", color="ChurnLabel", barmode="group",
        title="Active Membership Status vs Churn", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "08_active_vs_churn.png")
    figures["active_vs_churn"] = fig

    # 9. Credit Card vs Churn ---------------------------------------------------
    plot_df["CrCardLabel"] = plot_df["HasCrCard"].map({1: "Has Credit Card", 0: "No Credit Card"})
    cc_churn = plot_df.groupby(["CrCardLabel", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        cc_churn, x="CrCardLabel", y="Count", color="ChurnLabel", barmode="group",
        title="Credit Card Ownership vs Churn", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "09_creditcard_vs_churn.png")
    figures["creditcard_vs_churn"] = fig

    # 10. Tenure vs Churn -----------------------------------------------------
    tenure_churn = plot_df.groupby(["Tenure", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.line(
        tenure_churn, x="Tenure", y="Count", color="ChurnLabel", markers=True,
        title="Tenure vs Churn", color_discrete_map=CHURN_COLOR_MAP, template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "10_tenure_vs_churn.png")
    figures["tenure_vs_churn"] = fig

    # 11. Correlation Heatmap (Matplotlib) ---------------------------------------
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                     "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
                     "EngagementScore", "RelationshipStrengthIndex", "RetentionScore"]
    numeric_cols = [c for c in numeric_cols if c in plot_df.columns]
    corr = plot_df[numeric_cols].astype(float).corr()

    fig_mpl, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=7)
    ax.set_title("Correlation Heatmap - Key Numeric Features")
    fig_mpl.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_matplotlib(fig_mpl, "11_correlation_heatmap.png")

    # Also build an interactive Plotly version for the dashboard
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap - Key Numeric Features", template=PLOTLY_TEMPLATE,
    )
    figures["correlation_heatmap"] = fig

    # 12. Engagement Score Distribution --------------------------------------------
    fig = px.histogram(
        plot_df, x="EngagementScore", color="ChurnLabel", nbins=30, barmode="overlay",
        opacity=0.7, title="Engagement Score Distribution", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "12_engagement_score_distribution.png")
    figures["engagement_score_distribution"] = fig

    # 13. Relationship Strength Distribution ------------------------------------
    fig = px.histogram(
        plot_df, x="RelationshipStrengthIndex", nbins=30,
        title="Relationship Strength Index Distribution",
        color_discrete_sequence=[COLOR_PALETTE["secondary"]], template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "13_relationship_strength_distribution.png")
    figures["relationship_strength_distribution"] = fig

    # 14. Premium Customer Risk ----------------------------------------------------
    premium_risk = plot_df.groupby(["PremiumCustomer", "AtRiskPremiumCustomer"]).size().reset_index(name="Count")
    fig = px.bar(
        premium_risk, x="PremiumCustomer", y="Count", color="AtRiskPremiumCustomer", barmode="group",
        title="Premium Customer Risk Breakdown",
        color_discrete_map={"Yes": COLOR_PALETTE["danger"], "No": COLOR_PALETTE["success"]},
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "14_premium_customer_risk.png")
    figures["premium_customer_risk"] = fig

    # 15. Sticky Customer Analysis --------------------------------------------------
    sticky_churn = plot_df.groupby(["StickyCustomer", "ChurnLabel"]).size().reset_index(name="Count")
    fig = px.bar(
        sticky_churn, x="StickyCustomer", y="Count", color="ChurnLabel", barmode="group",
        title="Sticky Customer Analysis (Retention Behavior)",
        color_discrete_map=CHURN_COLOR_MAP, template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "15_sticky_customer_analysis.png")
    figures["sticky_customer_analysis"] = fig

    # 16. Customer Segmentation (Lifetime Segment) -----------------------------------
    segment_counts = plot_df["CustomerLifetimeSegment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    fig = px.bar(
        segment_counts, x="Segment", y="Count", color="Segment",
        title="Customer Segmentation by Lifetime Stage",
        color_discrete_sequence=px.colors.qualitative.Bold, template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "16_customer_segmentation.png")
    figures["customer_segmentation"] = fig

    # 17. Product Utilization ----------------------------------------------------------
    product_counts = plot_df["NumOfProducts"].value_counts().sort_index().reset_index()
    product_counts.columns = ["NumOfProducts", "Count"]
    fig = px.bar(
        product_counts, x="NumOfProducts", y="Count",
        title="Product Utilization Across Customer Base",
        color_discrete_sequence=[COLOR_PALETTE["primary"]], template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "17_product_utilization.png")
    figures["product_utilization"] = fig

    # 18. Salary vs Balance Scatter ---------------------------------------------------
    fig = px.scatter(
        plot_df, x="EstimatedSalary", y="Balance", color="ChurnLabel", opacity=0.5,
        title="Salary vs Balance (colored by Churn)", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "18_salary_vs_balance_scatter.png")
    figures["salary_vs_balance_scatter"] = fig

    # 19. Balance vs Churn Boxplot ------------------------------------------------------
    fig = px.box(
        plot_df, x="ChurnLabel", y="Balance", color="ChurnLabel",
        title="Balance Distribution by Churn Status", color_discrete_map=CHURN_COLOR_MAP,
        template=PLOTLY_TEMPLATE,
    )
    _save_plotly(fig, "19_balance_vs_churn_boxplot.png")
    figures["balance_vs_churn_boxplot"] = fig

    # 20. Executive KPI Dashboard Figure (Matplotlib multi-panel) ------------------------
    fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig_mpl.suptitle("Executive KPI Dashboard Overview", fontsize=16, fontweight="bold")

    # Panel A: Churn vs Retention donut
    churn_counts = plot_df["ChurnLabel"].value_counts()
    axes[0, 0].pie(
        churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
        colors=[CHURN_COLOR_MAP.get(lbl, COLOR_PALETTE["neutral"]) for lbl in churn_counts.index],
        wedgeprops={"width": 0.4},
    )
    axes[0, 0].set_title("Churn vs Retention")

    # Panel B: Avg Engagement by Geography
    geo_engagement = plot_df.groupby("Geography")["EngagementScore"].mean().sort_values()
    axes[0, 1].barh(geo_engagement.index, geo_engagement.values, color=COLOR_PALETTE["secondary"])
    axes[0, 1].set_title("Avg Engagement Score by Geography")
    axes[0, 1].set_xlabel("Engagement Score")

    # Panel C: Risk Level distribution
    risk_counts = plot_df["RiskLevel"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
    axes[1, 0].bar(risk_counts.index, risk_counts.values,
                    color=[COLOR_PALETTE["success"], COLOR_PALETTE["warning"], COLOR_PALETTE["danger"]])
    axes[1, 0].set_title("Customer Risk Level Distribution")
    axes[1, 0].set_ylabel("Number of Customers")

    # Panel D: Avg Retention Score by Lifetime Segment
    seg_retention = plot_df.groupby("CustomerLifetimeSegment")["RetentionScore"].mean()
    seg_order = ["New", "Growing", "Established", "Veteran"]
    seg_retention = seg_retention.reindex([s for s in seg_order if s in seg_retention.index])
    axes[1, 1].plot(seg_retention.index, seg_retention.values, marker="o",
                     color=COLOR_PALETTE["primary"], linewidth=2)
    axes[1, 1].set_title("Avg Retention Score by Lifetime Segment")
    axes[1, 1].set_ylabel("Retention Score")

    _save_matplotlib(fig_mpl, "20_executive_kpi_dashboard.png")

    _log("VISUALIZATION", f"Generated {len(figures) + 2} charts (2 Matplotlib-only, rest dual-saved).")

    return figures


# ==========================================================================
# 7. BUSINESS INSIGHTS
# ==========================================================================

def generate_business_insights(df: pd.DataFrame, kpis: Dict[str, float]) -> List[str]:
    """Automatically derive plain-English business insights from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe.
    kpis : Dict[str, float]
        Pre-computed KPI dictionary.

    Returns
    -------
    List[str]
        List of insight strings, printed and later exported to CSV.
    """
    _log_header("STAGE 7: BUSINESS INSIGHT GENERATION")

    insights: List[str] = []

    # Insight 1: Products vs churn
    churn_by_products = df.groupby("NumOfProducts")["Exited"].mean() * 100
    product_counts_n = df.groupby("NumOfProducts").size()
    if len(churn_by_products) > 1:
        best_product_count = churn_by_products.idxmin()
        worst_product_count = churn_by_products.idxmax()
        small_sample_note = (
            f" (note: only {product_counts_n[worst_product_count]} customers hold "
            f"{worst_product_count} products, so this figure should be read with caution)"
            if product_counts_n[worst_product_count] < 100 else ""
        )
        insights.append(
            f"Customers holding {best_product_count} product(s) churn the least "
            f"({churn_by_products.min():.1f}%), while those with {worst_product_count} "
            f"product(s) churn the most ({churn_by_products.max():.1f}%){small_sample_note}. "
            "Customers with 3+ products churn sharply more than those with 1-2, suggesting "
            "aggressive over-selling of products may itself be a churn signal rather than "
            "a retention lever."
        )

    # Insight 2: Inactive premium customers
    at_risk_premium_pct = (df["AtRiskPremiumCustomer"] == "Yes").mean() * 100
    insights.append(
        f"{at_risk_premium_pct:.1f}% of the customer base are 'At-Risk Premium Customers' "
        "-> high-value clients who are inactive or under-engaged. These customers "
        "represent a hidden revenue risk and should be prioritized for proactive outreach."
    )

    # Insight 3: High balance doesn't guarantee loyalty
    high_balance_churn = df.loc[df["BalanceCategory"] == "High Balance", "Exited"].mean() * 100
    overall_churn = kpis["Churn Rate (%)"]
    if high_balance_churn > overall_churn:
        insights.append(
            f"High-balance customers churn at {high_balance_churn:.1f}%, which is "
            f"HIGHER than the overall churn rate of {overall_churn:.1f}%. This confirms "
            "that a large balance alone does not guarantee loyalty -- engagement matters more."
        )
    else:
        insights.append(
            f"High-balance customers churn at {high_balance_churn:.1f}%, below the "
            f"overall churn rate of {overall_churn:.1f}%, suggesting balance size is "
            "a mild protective factor but should not be relied on in isolation."
        )

    # Insight 4: Credit card ownership and retention
    cc_stick = kpis["Credit Card Stickiness Score (%)"]
    non_cc_retention = (df.loc[df["HasCrCard"] == 0, "Exited"] == 0).mean() * 100
    gap = cc_stick - non_cc_retention
    if abs(gap) >= 2:
        stronger = "credit card holders" if gap > 0 else "customers without a credit card"
        insights.append(
            f"Customers with a credit card retain at {cc_stick:.1f}%, versus "
            f"{non_cc_retention:.1f}% for those without one -- {stronger} show "
            f"meaningfully stronger retention (a {abs(gap):.1f} point gap)."
        )
    else:
        insights.append(
            f"Credit card holders retain at {cc_stick:.1f}% versus {non_cc_retention:.1f}% "
            "for non-holders -- a negligible gap, suggesting credit card ownership alone "
            "is not a meaningful retention lever in this portfolio and should not be "
            "over-weighted in retention strategy."
        )

    # Insight 5: Relationship strength predicts churn
    corr_val = df[["RelationshipStrengthIndex", "Exited"]].astype(float).corr().iloc[0, 1]
    if corr_val < -0.05:
        insights.append(
            f"Relationship Strength Index correlates negatively with churn "
            f"(r = {corr_val:.2f}): customers with deeper, longer-standing "
            "relationships with the bank are demonstrably less likely to leave."
        )
    elif corr_val > 0.05:
        insights.append(
            f"Relationship Strength Index correlates positively with churn "
            f"(r = {corr_val:.2f}), a counter-intuitive signal suggesting that "
            "relationship depth alone (tenure + products + balance ratio) is not "
            "sufficient to explain churn in this portfolio -- engagement and "
            "activity status are likely stronger drivers."
        )
    else:
        insights.append(
            f"Relationship Strength Index shows only a weak correlation with churn "
            f"(r = {corr_val:.2f}), indicating that relationship depth alone does not "
            "strongly predict attrition -- other factors such as activity status and "
            "product count carry more explanatory power."
        )

    # Insight 6: Inactivity is the dominant churn driver
    active_churn = df.loc[df["IsActiveMember"] == 1, "Exited"].mean() * 100
    inactive_churn = df.loc[df["IsActiveMember"] == 0, "Exited"].mean() * 100
    insights.append(
        f"Inactive members churn at {inactive_churn:.1f}% versus {active_churn:.1f}% "
        "for active members -- reactivation campaigns targeting dormant customers "
        "could yield outsized retention gains."
    )

    # Insight 7: Geography variation
    geo_churn = df.groupby("Geography")["Exited"].mean() * 100
    top_geo = geo_churn.idxmax()
    insights.append(
        f"'{top_geo}' shows the highest regional churn rate at {geo_churn.max():.1f}%, "
        "indicating potential market-specific service, pricing, or competitive "
        "pressures warranting local investigation."
    )

    # Insight 8: Age and churn
    age_churn = df.groupby("AgeGroup", observed=True)["Exited"].mean() * 100
    if len(age_churn) > 0:
        oldest_bucket = age_churn.idxmax()
        insights.append(
            f"The '{oldest_bucket}' age segment exhibits the highest churn rate "
            f"({age_churn.max():.1f}%), suggesting retention offers and communication "
            "styles may need to be tailored by life stage."
        )

    # Insight 9: Salary/balance mismatch
    mismatch_rate = (df["SalaryBalanceMismatch"] == "Yes").mean() * 100 if "SalaryBalanceMismatch" in df.columns else 0
    if mismatch_rate > 0:
        insights.append(
            f"{mismatch_rate:.1f}% of customers show a salary-versus-balance mismatch "
            "(very high balance with low salary, or very high salary with low balance), "
            "which helps surface accounts whose financial commitment and cash behavior are misaligned."
        )

    # Insight 10: Behavioral segments
    if "EngagementProfile" in df.columns:
        profile_counts = df["EngagementProfile"].value_counts()
        if not profile_counts.empty:
            dominant_profile = profile_counts.idxmax()
            dominant_share = profile_counts.max() / len(df) * 100
            insights.append(
                f"The largest behavioral segment is '{dominant_profile}' at {dominant_share:.1f}% of the portfolio, "
                "making it the most practical starting point for targeted retention playbooks."
            )

    for i, insight in enumerate(insights, start=1):
        _log("INSIGHT", f"{i}. {insight}")

    return insights


# ==========================================================================
# 8. EXPORT FUNCTIONS
# ==========================================================================

def export_all(
    cleaned_df: pd.DataFrame,
    kpis: Dict[str, float],
    insights: List[str],
) -> None:
    """Persist the cleaned data, KPI report, and business summary to disk.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Fully engineered / cleaned dataframe.
    kpis : Dict[str, float]
        KPI dictionary.
    insights : List[str]
        List of generated business insights.
    """
    _log_header("STAGE 8: EXPORTING RESULTS")

    cleaned_path = os.path.join(DATA_DIR, "cleaned_customer_data.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    _log("EXPORT", f"Saved cleaned dataset -> {cleaned_path}")

    kpi_path = os.path.join(DATA_DIR, "kpi_report.csv")
    kpi_df = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
    kpi_df.to_csv(kpi_path, index=False)
    _log("EXPORT", f"Saved KPI report -> {kpi_path}")

    summary_path = os.path.join(DATA_DIR, "business_summary.csv")
    summary_df = pd.DataFrame({"Insight #": range(1, len(insights) + 1), "Business Insight": insights})
    summary_df.to_csv(summary_path, index=False)
    _log("EXPORT", f"Saved business summary -> {summary_path}")


# ==========================================================================
# 9. PIPELINE ORCHESTRATION
# ==========================================================================

def run_full_pipeline(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """Execute the entire analytics pipeline end-to-end.

    Parameters
    ----------
    csv_path : Optional[str]
        Optional explicit path to the raw CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], List[str]]
        The fully engineered dataframe, KPI dictionary, and insight list --
        these are the three artifacts consumed by the Streamlit dashboard.
    """
    raw_df = load_data(csv_path)
    validate_data(raw_df)
    cleaned_df = clean_data(raw_df)
    feature_df = engineer_features(cleaned_df)
    kpis = calculate_kpis(feature_df)
    generate_all_charts(feature_df)
    insights = generate_business_insights(feature_df, kpis)
    export_all(feature_df, kpis, insights)

    _log_header("PIPELINE COMPLETE")
    _log("STATUS", "All stages executed successfully. Outputs available in 'charts/' and 'processed_data/'.")

    return feature_df, kpis, insights


def print_pipeline_summary(
    df: pd.DataFrame, kpis: Dict[str, float], insights: List[str]
) -> None:
    """Print a clean, formatted summary table of the full pipeline run.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered dataframe.
    kpis : Dict[str, float]
        KPI dictionary.
    insights : List[str]
        Generated business insights.
    """
    bar = "=" * 60
    print(f"\n{bar}")
    print("  PIPELINE EXECUTION SUMMARY")
    print(bar)
    print(f"  {'Metric':<35} {'Value':>20}")
    print(f"  {'-'*35} {'-'*20}")
    print(f"  {'Total rows (final)' :<35} {len(df):>20,}")
    print(f"  {'Total columns (final)' :<35} {df.shape[1]:>20,}")
    print(f"  {'Engineered features' :<35} {'12':>20}")
    print(f"  {'KPIs computed' :<35} {len(kpis):>20}")
    print(f"  {'Business insights generated' :<35} {len(insights):>20}")
    print(f"  {'Charts exported' :<35} {'20':>20}")
    print(f"  {'Output directory' :<35} {'processed_data/':>20}")
    print(f"  {'Charts directory' :<35} {'charts/':>20}")
    print(bar)


if __name__ == "__main__":
    _df, _kpis, _insights = run_full_pipeline()
    print_pipeline_summary(_df, _kpis, _insights)
