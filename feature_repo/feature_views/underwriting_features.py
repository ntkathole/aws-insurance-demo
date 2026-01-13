"""
Underwriting Feature Views - Real-Time Auto Underwriting (PCM)

These feature views support real-time policy underwriting and pricing decisions.
They are optimized for low-latency online serving via DynamoDB.

Transaction Type: Real-Time (PCM - Policy Calculation Module)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Float64, Int32, Int64, String, Bool

# Import entities and data sources
import sys
sys.path.insert(0, '..')
from entities import customer, policy
from data_sources import (
    customer_profile_source,
    customer_credit_source,
    customer_risk_source,
    customer_risk_push_source,
    policy_source,
)


# =============================================================================
# CUSTOMER PROFILE FEATURES
# =============================================================================
# Demographics and basic customer information for underwriting
customer_profile_fv = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=timedelta(days=1),  # Refresh daily
    schema=[
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
    ],
    online=True,
    source=customer_profile_source,
    tags={
        "domain": "underwriting",
        "use_case": "pcm",
        "latency_requirement": "low",
    },
    description="Customer demographic and profile features for underwriting",
)


# =============================================================================
# CUSTOMER CREDIT FEATURES
# =============================================================================
# Credit and financial indicators - updated more frequently
customer_credit_fv = FeatureView(
    name="customer_credit_features",
    entities=[customer],
    ttl=timedelta(hours=1),  # Refresh hourly - credit data changes
    schema=[
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
    ],
    online=True,
    source=customer_credit_source,
    tags={
        "domain": "underwriting",
        "use_case": "pcm",
        "latency_requirement": "low",
        "sensitive": "true",
    },
    description="Customer credit and financial features for risk assessment",
)


# =============================================================================
# CUSTOMER RISK FEATURES
# =============================================================================
# Risk metrics and behavioral patterns
customer_risk_fv = FeatureView(
    name="customer_risk_features",
    entities=[customer],
    ttl=timedelta(hours=1),
    schema=[
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
    source=customer_risk_source,
    tags={
        "domain": "underwriting",
        "use_case": "pcm",
        "latency_requirement": "low",
    },
    description="Customer risk assessment features for pricing and decisioning",
)


# =============================================================================
# CUSTOMER RISK FEATURES (FRESH) - Push Source for Real-Time Updates
# =============================================================================
# Same schema as above but using push source for real-time updates
customer_risk_fresh_fv = FeatureView(
    name="customer_risk_features_fresh",
    entities=[customer],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="overall_risk_score", dtype=Float32),
        Field(name="claims_risk_score", dtype=Float32),
        Field(name="fraud_risk_score", dtype=Float32),
        Field(name="churn_risk_score", dtype=Float32),
        Field(name="num_claims_1y", dtype=Int32),
        Field(name="num_claims_3y", dtype=Int32),
        Field(name="total_claims_amount_1y", dtype=Float64),
        Field(name="avg_claim_amount", dtype=Float64),
        Field(name="policy_changes_1y", dtype=Int32),
        Field(name="late_payments_1y", dtype=Int32),
        Field(name="inquiry_count_30d", dtype=Int32),
        Field(name="driving_violations_3y", dtype=Int32),
        Field(name="at_fault_accidents_3y", dtype=Int32),
        Field(name="dui_flag", dtype=Bool),
        Field(name="risk_segment", dtype=String),
        Field(name="underwriting_tier", dtype=String),
    ],
    online=True,
    source=customer_risk_push_source,  # Push source for real-time updates
    tags={
        "domain": "underwriting",
        "use_case": "pcm",
        "latency_requirement": "real_time",
        "fresh": "true",
    },
    description="Fresh customer risk features via push source",
)


# =============================================================================
# POLICY FEATURES
# =============================================================================
# Policy-level features for coverage analysis
policy_fv = FeatureView(
    name="policy_features",
    entities=[policy],
    ttl=timedelta(days=1),
    schema=[
        # Policy identification
        Field(name="customer_id", dtype=String, description="Associated customer ID"),
        Field(name="policy_type", dtype=String, description="Type of policy (auto/home/life)"),
        Field(name="product_code", dtype=String, description="Product code"),
        
        # Coverage details
        Field(name="coverage_amount", dtype=Float64, description="Total coverage amount"),
        Field(name="deductible", dtype=Float64, description="Deductible amount"),
        Field(name="premium_monthly", dtype=Float64, description="Monthly premium"),
        Field(name="premium_annual", dtype=Float64, description="Annual premium"),
        
        # Policy dates
        Field(name="policy_start_date", dtype=String, description="Policy start date"),
        Field(name="policy_term_months", dtype=Int32, description="Policy term in months"),
        Field(name="days_until_renewal", dtype=Int32, description="Days until renewal"),
        
        # Coverage options
        Field(name="has_collision", dtype=Bool, description="Has collision coverage"),
        Field(name="has_comprehensive", dtype=Bool, description="Has comprehensive coverage"),
        Field(name="has_liability", dtype=Bool, description="Has liability coverage"),
        Field(name="has_uninsured_motorist", dtype=Bool, description="Has uninsured motorist"),
        Field(name="has_roadside", dtype=Bool, description="Has roadside assistance"),
        
        # Discounts
        Field(name="multi_policy_discount", dtype=Bool, description="Has multi-policy discount"),
        Field(name="safe_driver_discount", dtype=Bool, description="Has safe driver discount"),
        Field(name="paperless_discount", dtype=Bool, description="Has paperless discount"),
        Field(name="total_discount_pct", dtype=Float32, description="Total discount percentage"),
        
        # Vehicle/Property info (for auto/home)
        Field(name="vehicle_year", dtype=Int32, description="Vehicle year"),
        Field(name="vehicle_make", dtype=String, description="Vehicle make"),
        Field(name="vehicle_value", dtype=Float64, description="Vehicle/property value"),
    ],
    online=True,
    source=policy_source,
    tags={
        "domain": "underwriting",
        "use_case": "pcm",
    },
    description="Policy details and coverage information",
)
