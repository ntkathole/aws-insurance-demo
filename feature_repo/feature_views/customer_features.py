"""
Consolidated Customer Feature View - Simplified Architecture

This feature view consolidates customer profile, credit, and risk features into a single wide view
with 40+ columns for comprehensive latency testing. This supports the simplified architecture with:
- 1 batch feature view (this one) for variable feature count testing
- 1 on-demand feature view (premium calculator) for transformation testing

Total Features: 40 (13 profile + 11 credit + 16 risk)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Float64, Int32, Int64, String, Bool

# Import entities and data sources
import sys
sys.path.insert(0, '..')
from entities import customer
from data_sources import customer_consolidated_source


# =============================================================================
# CONSOLIDATED CUSTOMER FEATURE VIEW - All Customer Features in One Place
# =============================================================================
customer_consolidated_fv = FeatureView(
    name="customer_consolidated_features",
    entities=[customer],
    ttl=timedelta(hours=1),  # Refresh hourly to match most frequent source
    schema=[
        # ===== PROFILE FEATURES (13) =====
        # Demographics
        Field(name="age", dtype=Int32, description="Customer age"),
        Field(name="gender", dtype=String, description="Customer gender"),
        Field(name="marital_status", dtype=String, description="Marital status"),
        Field(name="occupation", dtype=String, description="Occupation category"),
        Field(name="education_level", dtype=String, description="Education level"),

        # Location
        Field(name="state", dtype=String, description="State of residence"),
        Field(name="zip_code", dtype=String, description="ZIP code"),
        Field(name="urban_rural", dtype=String, description="Urban/Suburban/Rural"),
        Field(name="region_risk_zone", dtype=Int32, description="Regional risk zone (1-5)"),

        # Customer history
        Field(name="customer_tenure_months", dtype=Int32, description="Months as customer"),
        Field(name="num_policies", dtype=Int32, description="Number of active policies"),
        Field(name="loyalty_tier", dtype=String, description="Customer loyalty tier"),
        Field(name="has_agent", dtype=Bool, description="Has assigned agent"),

        # ===== CREDIT FEATURES (11) =====
        # Credit scores
        Field(name="credit_score", dtype=Int32, description="FICO credit score"),
        Field(name="credit_score_tier", dtype=String, description="Credit tier (A-F)"),
        Field(name="credit_score_change_3m", dtype=Int32, description="Credit score change in 3 months"),

        # Credit history
        Field(name="credit_history_length_months", dtype=Int32, description="Length of credit history"),
        Field(name="num_credit_accounts", dtype=Int32, description="Number of credit accounts"),
        Field(name="num_delinquencies", dtype=Int32, description="Number of delinquencies"),
        Field(name="bankruptcy_flag", dtype=Bool, description="Has bankruptcy on record"),

        # Financial indicators
        Field(name="annual_income", dtype=Float64, description="Estimated annual income"),
        Field(name="debt_to_income_ratio", dtype=Float32, description="Debt to income ratio"),
        Field(name="payment_history_score", dtype=Float32, description="Payment history score (0-100)"),

        # Insurance-specific credit factors
        Field(name="insurance_score", dtype=Int32, description="Insurance-specific credit score"),
        Field(name="prior_coverage_lapse", dtype=Bool, description="Had coverage lapse"),

        # ===== RISK FEATURES (16) =====
        # Risk scores
        Field(name="overall_risk_score", dtype=Float32, description="Overall risk score (0-100)"),
        Field(name="claims_risk_score", dtype=Float32, description="Claims propensity score"),
        Field(name="fraud_risk_score", dtype=Float32, description="Fraud risk indicator"),
        Field(name="churn_risk_score", dtype=Float32, description="Churn probability"),

        # Historical metrics
        Field(name="num_claims_1y", dtype=Int32, description="Number of claims in last year"),
        Field(name="num_claims_3y", dtype=Int32, description="Number of claims in last 3 years"),
        Field(name="total_claims_amount_1y", dtype=Float64, description="Total claims amount in last year"),
        Field(name="avg_claim_amount", dtype=Float64, description="Average claim amount"),

        # Behavioral indicators
        Field(name="policy_changes_1y", dtype=Int32, description="Policy changes in last year"),
        Field(name="late_payments_1y", dtype=Int32, description="Late payments in last year"),
        Field(name="inquiry_count_30d", dtype=Int32, description="Inquiries in last 30 days"),

        # Risk factors (auto insurance specific)
        Field(name="driving_violations_3y", dtype=Int32, description="Driving violations in 3 years"),
        Field(name="at_fault_accidents_3y", dtype=Int32, description="At-fault accidents in 3 years"),
        Field(name="dui_flag", dtype=Bool, description="DUI on record"),

        # Risk segments
        Field(name="risk_segment", dtype=String, description="Risk segment (low/medium/high)"),
        Field(name="underwriting_tier", dtype=String, description="Underwriting tier"),
    ],
    online=True,
    source=customer_consolidated_source,
    tags={
        "domain": "customer",
        "consolidated": "true",
        "use_case": "latency_testing",
        "total_features": "40",
        "latency_requirement": "low",
    },
    description="Consolidated customer feature view with 40+ features for latency testing",
)


# =============================================================================
# FEATURE GROUPINGS FOR LATENCY TESTING
# =============================================================================

# Feature groups for variable latency testing
FEATURE_GROUPS = {
    "basic_5": [
        "age", "gender", "state", "credit_score", "overall_risk_score"
    ],
    "standard_10": [
        "age", "gender", "state", "region_risk_zone", "credit_score",
        "insurance_score", "overall_risk_score", "claims_risk_score",
        "num_claims_3y", "customer_tenure_months"
    ],
    "comprehensive_20": [
        "age", "gender", "marital_status", "occupation", "state",
        "region_risk_zone", "customer_tenure_months", "num_policies",
        "credit_score", "credit_score_tier", "insurance_score", "bankruptcy_flag",
        "overall_risk_score", "claims_risk_score", "fraud_risk_score",
        "num_claims_3y", "avg_claim_amount", "late_payments_1y",
        "risk_segment", "underwriting_tier"
    ],
    "all_40": None  # All features (don't specify subset)
}

# Profile features subset (13 features)
PROFILE_FEATURES = [
    "age", "gender", "marital_status", "occupation", "education_level",
    "state", "zip_code", "urban_rural", "region_risk_zone",
    "customer_tenure_months", "num_policies", "loyalty_tier", "has_agent"
]

# Credit features subset (11 features)
CREDIT_FEATURES = [
    "credit_score", "credit_score_tier", "credit_score_change_3m",
    "credit_history_length_months", "num_credit_accounts", "num_delinquencies",
    "bankruptcy_flag", "annual_income", "debt_to_income_ratio",
    "payment_history_score", "insurance_score", "prior_coverage_lapse"
]

# Risk features subset (16 features)
RISK_FEATURES = [
    "overall_risk_score", "claims_risk_score", "fraud_risk_score", "churn_risk_score",
    "num_claims_1y", "num_claims_3y", "total_claims_amount_1y", "avg_claim_amount",
    "policy_changes_1y", "late_payments_1y", "inquiry_count_30d",
    "driving_violations_3y", "at_fault_accidents_3y", "dui_flag",
    "risk_segment", "underwriting_tier"
]