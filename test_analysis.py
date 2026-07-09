"""
==============================================================================
 TEST SUITE - Customer Engagement & Retention Analytics Pipeline
==============================================================================

Lightweight pytest suite covering every stage of the analytics pipeline.
Run with:  pytest test_analysis.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis as az


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the raw dataset once for the entire test module."""
    return az.load_data()


@pytest.fixture(scope="module")
def validated_report(raw_df) -> dict:
    """Run validation and return the report dict."""
    return az.validate_data(raw_df)


@pytest.fixture(scope="module")
def cleaned_df(raw_df) -> pd.DataFrame:
    """Return the cleaned dataframe."""
    return az.clean_data(raw_df)


@pytest.fixture(scope="module")
def engineered_df(cleaned_df) -> pd.DataFrame:
    """Return the feature-engineered dataframe."""
    return az.engineer_features(cleaned_df)


@pytest.fixture(scope="module")
def kpis(engineered_df) -> dict:
    """Return the computed KPI dictionary."""
    return az.calculate_kpis(engineered_df)


@pytest.fixture(scope="module")
def insights(engineered_df, kpis) -> list:
    """Return the generated business insights."""
    return az.generate_business_insights(engineered_df, kpis)


# ---------------------------------------------------------------------------
# STAGE 1: DATA LOADING
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests for data loading and column detection."""

    def test_load_returns_dataframe(self, raw_df):
        """load_data() should return a pandas DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)

    def test_load_has_rows(self, raw_df):
        """Loaded dataset should not be empty."""
        assert len(raw_df) > 0

    def test_required_columns_present(self, raw_df):
        """All minimum required columns should be present after auto-detection."""
        for col in az.MINIMUM_REQUIRED_COLUMNS:
            assert col in raw_df.columns, f"Missing required column: {col}"

    def test_load_missing_file_raises(self):
        """load_data() with a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            az.load_data("/nonexistent/path/data.csv")

    def test_column_detection_maps_known_aliases(self):
        """detect_columns should map known column aliases correctly."""
        test_df = pd.DataFrame({"credit_score": [1], "age": [30], "churn": [0]})
        mapping = az.detect_columns(test_df)
        assert "CreditScore" in mapping
        assert "Age" in mapping
        assert "Exited" in mapping


# ---------------------------------------------------------------------------
# STAGE 2: DATA VALIDATION
# ---------------------------------------------------------------------------

class TestDataValidation:
    """Tests for the data quality audit."""

    def test_returns_dict(self, validated_report):
        """validate_data() should return a dictionary."""
        assert isinstance(validated_report, dict)

    def test_expected_keys(self, validated_report):
        """Report should contain expected quality-check keys."""
        expected_keys = ["missing_values", "duplicate_rows"]
        for key in expected_keys:
            assert key in validated_report, f"Missing validation key: {key}"

    def test_duplicate_rows_is_int(self, validated_report):
        """Duplicate row count should be an integer."""
        assert isinstance(validated_report["duplicate_rows"], int)


# ---------------------------------------------------------------------------
# STAGE 3: DATA CLEANING
# ---------------------------------------------------------------------------

class TestDataCleaning:
    """Tests for the cleaning stage."""

    def test_returns_dataframe(self, cleaned_df):
        """clean_data() should return a DataFrame."""
        assert isinstance(cleaned_df, pd.DataFrame)

    def test_no_duplicates(self, cleaned_df):
        """Cleaned data should have no fully duplicated rows."""
        assert cleaned_df.duplicated().sum() == 0

    def test_no_missing_values(self, cleaned_df):
        """Cleaned data should have no missing values in critical columns."""
        for col in ["CreditScore", "Age", "Balance", "Exited"]:
            if col in cleaned_df.columns:
                assert cleaned_df[col].isnull().sum() == 0, f"Missing values in {col}"

    def test_age_range_valid(self, cleaned_df):
        """All ages should be within [18, 100] after cleaning."""
        assert cleaned_df["Age"].min() >= 18
        assert cleaned_df["Age"].max() <= 100

    def test_no_negative_balances(self, cleaned_df):
        """No negative balances should remain after cleaning."""
        assert (cleaned_df["Balance"] >= 0).all()

    def test_products_in_valid_range(self, cleaned_df):
        """Product counts should be in [1, 4] after cleaning."""
        assert cleaned_df["NumOfProducts"].min() >= 1
        assert cleaned_df["NumOfProducts"].max() <= 4


# ---------------------------------------------------------------------------
# STAGE 4: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for derived feature creation."""

    EXPECTED_FEATURES = [
        "EngagementScore",
        "EngagementTier",
        "RelationshipStrengthIndex",
        "ProductDepthIndex",
        "PremiumCustomer",
        "SalaryBalanceMismatch",
        "SalaryBalanceMismatchType",
        "StickyCustomer",
        "AtRiskPremiumCustomer",
        "EngagementProfile",
        "CustomerLifetimeSegment",
        "AgeGroup",
        "BalanceCategory",
        "SalaryCategory",
        "RiskLevel",
        "RetentionScore",
        "RetentionStabilityTier",
    ]

    def test_all_features_created(self, engineered_df):
        """All 12 engineered features should be present in the dataframe."""
        for feat in self.EXPECTED_FEATURES:
            assert feat in engineered_df.columns, f"Missing feature: {feat}"

    def test_engagement_score_range(self, engineered_df):
        """EngagementScore should be in [0, 100]."""
        assert engineered_df["EngagementScore"].min() >= 0
        assert engineered_df["EngagementScore"].max() <= 100

    def test_retention_score_range(self, engineered_df):
        """RetentionScore should be in [0, 100]."""
        assert engineered_df["RetentionScore"].min() >= 0
        assert engineered_df["RetentionScore"].max() <= 100

    def test_premium_customer_binary(self, engineered_df):
        """PremiumCustomer should only contain 'Yes' or 'No'."""
        assert set(engineered_df["PremiumCustomer"].unique()) <= {"Yes", "No"}

    def test_risk_level_categories(self, engineered_df):
        """RiskLevel should only contain expected categories."""
        valid = {"Low Risk", "Medium Risk", "High Risk"}
        actual = set(engineered_df["RiskLevel"].dropna().unique())
        assert actual <= valid, f"Unexpected risk levels: {actual - valid}"

    def test_product_depth_index_range(self, engineered_df):
        """ProductDepthIndex should be in [0, 1]."""
        assert engineered_df["ProductDepthIndex"].min() >= 0
        assert engineered_df["ProductDepthIndex"].max() <= 1

    def test_engagement_tier_categories(self, engineered_df):
        """EngagementTier should only contain Low/Medium/High."""
        valid = {"Low", "Medium", "High"}
        actual = set(engineered_df["EngagementTier"].dropna().astype(str).unique())
        assert actual <= valid

    def test_engagement_profile_categories(self, engineered_df):
        """EngagementProfile should use the four required labels plus Other."""
        valid = {
            "Active Engaged",
            "Inactive Disengaged",
            "Active but Low-Product",
            "Inactive High-Balance",
            "Other",
        }
        actual = set(engineered_df["EngagementProfile"].dropna().astype(str).unique())
        assert actual <= valid

    def test_retention_stability_tier_categories(self, engineered_df):
        """RetentionStabilityTier should only contain Sticky, Stable, or Watchlist."""
        valid = {"Sticky", "Stable", "Watchlist"}
        actual = set(engineered_df["RetentionStabilityTier"].dropna().astype(str).unique())
        assert actual <= valid

    def test_salary_balance_mismatch_categories(self, engineered_df):
        """SalaryBalanceMismatch should be Yes/No and the type should be explicit."""
        assert set(engineered_df["SalaryBalanceMismatch"].unique()) <= {"Yes", "No"}
        valid = {"None", "High Balance / Low Salary", "Low Balance / High Salary"}
        actual = set(engineered_df["SalaryBalanceMismatchType"].dropna().astype(str).unique())
        assert actual <= valid


# ---------------------------------------------------------------------------
# STAGE 5: KPI CALCULATIONS
# ---------------------------------------------------------------------------

class TestKPICalculations:
    """Tests for KPI computation."""

    EXPECTED_KPIS = [
        "Total Customers",
        "Total Churn",
        "Churn Rate (%)",
        "Retention Rate (%)",
        "Average Balance",
        "Average Salary",
        "Average Products",
        "Average Credit Score",
        "Active Customer (%)",
        "Inactive Customer (%)",
        "Premium Customer (%)",
        "Product Utilization Index",
        "Relationship Strength Index",
        "Engagement Retention Ratio",
        "High Balance Disengagement Rate (%)",
        "Credit Card Stickiness Score (%)",
        "Average Tenure (yrs)",
    ]

    def test_returns_dict(self, kpis):
        """calculate_kpis() should return a dictionary."""
        assert isinstance(kpis, dict)

    def test_all_kpis_present(self, kpis):
        """All 17 expected KPIs should be present."""
        for kpi_name in self.EXPECTED_KPIS:
            assert kpi_name in kpis, f"Missing KPI: {kpi_name}"

    def test_kpi_count(self, kpis):
        """Should have exactly 17 KPIs."""
        assert len(kpis) == 17

    def test_churn_retention_sum(self, kpis):
        """Churn rate + Retention rate should equal 100%."""
        total = kpis["Churn Rate (%)"] + kpis["Retention Rate (%)"]
        assert abs(total - 100.0) < 0.1, f"Churn + Retention = {total}, expected ~100"

    def test_active_inactive_sum(self, kpis):
        """Active + Inactive percentages should equal 100%."""
        total = kpis["Active Customer (%)"] + kpis["Inactive Customer (%)"]
        assert abs(total - 100.0) < 0.1, f"Active + Inactive = {total}, expected ~100"

    def test_no_nan_kpis(self, kpis):
        """No KPI should be NaN (except Engagement Retention Ratio in edge cases)."""
        for name, value in kpis.items():
            if name == "Engagement Retention Ratio":
                continue  # Can be NaN if no churned customers in filter
            assert not (isinstance(value, float) and np.isnan(value)), (
                f"KPI '{name}' is NaN"
            )

    def test_percentages_in_range(self, kpis):
        """Percentage KPIs should be between 0 and 100."""
        pct_kpis = [
            "Churn Rate (%)", "Retention Rate (%)", "Active Customer (%)",
            "Inactive Customer (%)", "Premium Customer (%)",
        ]
        for name in pct_kpis:
            assert 0 <= kpis[name] <= 100, f"{name} = {kpis[name]} is out of [0, 100]"


# ---------------------------------------------------------------------------
# STAGE 7: BUSINESS INSIGHTS
# ---------------------------------------------------------------------------

class TestBusinessInsights:
    """Tests for automated insight generation."""

    def test_returns_list(self, insights):
        """generate_business_insights() should return a list."""
        assert isinstance(insights, list)

    def test_non_empty(self, insights):
        """Should generate at least one insight."""
        assert len(insights) > 0

    def test_all_strings(self, insights):
        """Every insight should be a string."""
        for i, insight in enumerate(insights):
            assert isinstance(insight, str), f"Insight #{i+1} is not a string"

    def test_insights_are_substantive(self, insights):
        """Each insight should be at least 50 characters (not trivial)."""
        for i, insight in enumerate(insights):
            assert len(insight) >= 50, (
                f"Insight #{i+1} is too short ({len(insight)} chars): {insight[:40]}..."
            )

    def test_minimum_insight_count(self, insights):
        """Should generate at least 5 insights for a typical dataset."""
        assert len(insights) >= 5


# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Tests for module-level configuration."""

    def test_color_palette_keys(self):
        """COLOR_PALETTE should have all expected keys."""
        expected = ["primary", "secondary", "accent", "success", "danger",
                    "warning", "neutral", "background"]
        for key in expected:
            assert key in az.COLOR_PALETTE, f"Missing palette key: {key}"

    def test_churn_color_map(self):
        """CHURN_COLOR_MAP should have Retained and Churned entries."""
        assert "Retained" in az.CHURN_COLOR_MAP
        assert "Churned" in az.CHURN_COLOR_MAP

    def test_plotly_template(self):
        """PLOTLY_TEMPLATE should be a valid Plotly template name."""
        assert az.PLOTLY_TEMPLATE in ["plotly_white", "plotly_dark", "plotly", "simple_white"]

    def test_public_api_exports(self):
        """__all__ should be defined and contain key functions."""
        assert hasattr(az, "__all__")
        assert "run_full_pipeline" in az.__all__
        assert "calculate_kpis" in az.__all__


# ---------------------------------------------------------------------------
# INTEGRATION
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration sanity checks."""

    def test_pipeline_output_consistency(self, engineered_df, kpis):
        """KPI 'Total Customers' should match the dataframe row count."""
        assert kpis["Total Customers"] == len(engineered_df)

    def test_churn_label_mapping(self, engineered_df):
        """_map_churn_labels should correctly map 0->Retained, 1->Churned."""
        labels = az._map_churn_labels(engineered_df["Exited"])
        assert set(labels.unique()) <= {"Retained", "Churned"}

    def test_export_paths_exist(self):
        """Output directories should exist (created at module import)."""
        assert os.path.isdir(az.DATA_DIR)
        assert os.path.isdir(az.CHARTS_DIR)
