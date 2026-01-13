"""
On-Demand Feature Views for AWS Insurance Demo.

On-demand feature views enable real-time feature transformations that combine:
- Existing features from feature views
- Request-time data (from API requests)
- Business logic transformations

These are critical for:
- Real-time risk scoring (underwriting)
- Fraud detection signals (claims)
- Dynamic premium calculations
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from feast import Field, RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int32, Int64, String, Bool

# Import feature views
from feature_views.underwriting_features import (
    customer_profile_fv,
    customer_credit_fv,
    customer_risk_fv,
    policy_fv,
)
from feature_views.claims_features import (
    claims_history_fv,
    claims_aggregation_fv,
    lab_results_fv,
    provider_fv,
)
from feature_views.streaming_features import (
    transaction_features_fv,
)


# =============================================================================
# REQUEST SOURCES - Runtime Inputs
# =============================================================================

# Request source for underwriting - inputs at quote time
underwriting_request = RequestSource(
    name="underwriting_request",
    schema=[
        Field(name="requested_coverage", dtype=Float64, description="Requested coverage amount"),
        Field(name="requested_deductible", dtype=Float64, description="Requested deductible"),
        Field(name="policy_type", dtype=String, description="Type of policy requested"),
        Field(name="term_months", dtype=Int32, description="Requested term in months"),
        Field(name="additional_drivers", dtype=Int32, description="Number of additional drivers"),
        Field(name="vehicle_age", dtype=Int32, description="Age of vehicle in years"),
    ],
    description="Request-time inputs for underwriting decisions",
)

# Request source for fraud detection - transaction context
fraud_detection_request = RequestSource(
    name="fraud_detection_request",
    schema=[
        Field(name="transaction_amount", dtype=Float64, description="Current transaction amount"),
        Field(name="merchant_category", dtype=String, description="Merchant category code"),
        Field(name="transaction_channel", dtype=String, description="Transaction channel"),
        Field(name="device_trust_score", dtype=Float32, description="Device trust score"),
        Field(name="session_duration_seconds", dtype=Int32, description="Session duration"),
        Field(name="is_international", dtype=Bool, description="Is international transaction"),
    ],
    description="Request-time inputs for fraud detection",
)

# Request source for claims - claim context
claims_assessment_request = RequestSource(
    name="claims_assessment_request",
    schema=[
        Field(name="claim_amount_requested", dtype=Float64, description="Claim amount requested"),
        Field(name="claim_type", dtype=String, description="Type of claim"),
        Field(name="days_since_incident", dtype=Int32, description="Days since incident"),
        Field(name="documentation_score", dtype=Float32, description="Documentation completeness"),
        Field(name="has_witnesses", dtype=Bool, description="Has witnesses"),
        Field(name="police_report_filed", dtype=Bool, description="Police report filed"),
    ],
    description="Request-time inputs for claims assessment",
)


# =============================================================================
# ON-DEMAND FEATURE VIEWS - Underwriting (Real-Time PCM)
# =============================================================================

@on_demand_feature_view(
    sources=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_request,
    ],
    schema=[
        Field(name="composite_risk_score", dtype=Float64, description="Composite risk score"),
        Field(name="risk_tier", dtype=String, description="Risk tier (preferred/standard/substandard)"),
        Field(name="base_premium_factor", dtype=Float64, description="Base premium factor"),
        Field(name="credit_adjustment_factor", dtype=Float64, description="Credit adjustment factor"),
        Field(name="experience_adjustment_factor", dtype=Float64, description="Experience adjustment factor"),
        Field(name="coverage_adjustment_factor", dtype=Float64, description="Coverage adjustment factor"),
        Field(name="final_premium_factor", dtype=Float64, description="Final premium multiplier"),
        Field(name="underwriting_decision", dtype=String, description="Auto-underwriting decision"),
        Field(name="manual_review_flag", dtype=Bool, description="Requires manual review"),
        Field(name="decline_reason", dtype=String, description="Reason for decline if applicable"),
    ],
    mode="pandas",
    description="Real-time underwriting risk score calculation",
    tags={"domain": "underwriting", "use_case": "pcm", "real_time": "true"},
)
def underwriting_risk_score(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate real-time underwriting risk score and premium factors.
    
    This combines customer profile, credit, risk features with request-time
    inputs to produce underwriting decisions and premium calculations.
    """
    df = pd.DataFrame()
    
    # Calculate composite risk score (weighted average of risk factors)
    # Weights: credit 30%, claims history 25%, profile 20%, behavioral 25%
    credit_factor = (inputs["credit_score"].fillna(650) - 300) / 550  # Normalize 0-1
    claims_factor = 1 - (inputs["overall_risk_score"].fillna(50) / 100)  # Inverse
    profile_factor = (inputs["customer_tenure_months"].fillna(0) / 120).clip(0, 1)
    behavioral_factor = 1 - (inputs["late_payments_1y"].fillna(0) / 12).clip(0, 1)
    
    df["composite_risk_score"] = (
        credit_factor * 0.30 +
        claims_factor * 0.25 +
        profile_factor * 0.20 +
        behavioral_factor * 0.25
    ) * 100  # Scale to 0-100
    
    # Determine risk tier
    df["risk_tier"] = pd.cut(
        df["composite_risk_score"],
        bins=[0, 40, 60, 80, 100],
        labels=["substandard", "standard", "preferred", "super_preferred"],
    ).astype(str)
    
    # Calculate premium factors
    # Base factor from risk score
    df["base_premium_factor"] = 2.0 - (df["composite_risk_score"] / 100)
    
    # Credit adjustment
    df["credit_adjustment_factor"] = (
        inputs["insurance_score"].fillna(700).apply(
            lambda x: 0.9 if x > 800 else (1.0 if x > 700 else (1.1 if x > 600 else 1.3))
        )
    )
    
    # Experience adjustment (claims history)
    df["experience_adjustment_factor"] = (
        1.0 + (inputs["num_claims_3y"].fillna(0) * 0.1)
    ).clip(1.0, 1.5)
    
    # Coverage adjustment
    coverage_ratio = inputs["requested_coverage"] / inputs["requested_coverage"].median()
    df["coverage_adjustment_factor"] = (1.0 + (coverage_ratio - 1) * 0.1).clip(0.8, 1.5)
    
    # Final premium factor
    df["final_premium_factor"] = (
        df["base_premium_factor"] *
        df["credit_adjustment_factor"] *
        df["experience_adjustment_factor"] *
        df["coverage_adjustment_factor"]
    ).round(4)
    
    # Underwriting decision
    conditions = [
        (df["composite_risk_score"] >= 70) & (inputs["bankruptcy_flag"].fillna(False) == False),
        (df["composite_risk_score"] >= 50) & (df["composite_risk_score"] < 70),
        (df["composite_risk_score"] >= 30) & (df["composite_risk_score"] < 50),
    ]
    choices = ["auto_approve", "auto_approve_with_conditions", "manual_review"]
    df["underwriting_decision"] = pd.Series(
        np.select(conditions, choices, default="decline"),
        index=df.index
    )
    
    # Manual review flag
    df["manual_review_flag"] = (
        (df["underwriting_decision"] == "manual_review") |
        (inputs["dui_flag"].fillna(False)) |
        (inputs["num_claims_3y"].fillna(0) > 3) |
        (inputs["requested_coverage"].fillna(0) > 500000)
    )
    
    # Decline reason
    decline_reasons = []
    for idx, row in inputs.iterrows():
        reasons = []
        if row.get("dui_flag", False):
            reasons.append("DUI_ON_RECORD")
        if row.get("bankruptcy_flag", False):
            reasons.append("RECENT_BANKRUPTCY")
        if row.get("num_claims_3y", 0) > 5:
            reasons.append("EXCESSIVE_CLAIMS")
        if df.loc[idx, "composite_risk_score"] < 30:
            reasons.append("HIGH_RISK_SCORE")
        decline_reasons.append(";".join(reasons) if reasons else "")
    df["decline_reason"] = decline_reasons
    
    return df


@on_demand_feature_view(
    sources=[
        customer_profile_fv,
        customer_credit_fv,
        underwriting_request,
    ],
    schema=[
        Field(name="age_factor", dtype=Float64),
        Field(name="location_factor", dtype=Float64),
        Field(name="vehicle_age_factor", dtype=Float64),
        Field(name="coverage_factor", dtype=Float64),
        Field(name="deductible_credit", dtype=Float64),
        Field(name="estimated_base_premium", dtype=Float64),
        Field(name="estimated_monthly_premium", dtype=Float64),
        Field(name="estimated_annual_premium", dtype=Float64),
    ],
    mode="pandas",
    description="Dynamic premium calculation based on risk factors",
    tags={"domain": "underwriting", "use_case": "pcm", "real_time": "true"},
)
def premium_calculator(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate estimated premium based on customer profile and request parameters.
    """
    df = pd.DataFrame()
    
    # Age factor (drivers 25-65 get best rates)
    age = inputs["age"].fillna(35)
    df["age_factor"] = np.select(
        [age < 25, age < 30, age <= 65, age <= 75],
        [1.4, 1.1, 1.0, 1.2],
        default=1.5
    )
    
    # Location factor based on region risk zone
    region_risk = inputs["region_risk_zone"].fillna(3)
    df["location_factor"] = region_risk.map({1: 0.85, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.35}).fillna(1.0)
    
    # Vehicle age factor
    vehicle_age = inputs["vehicle_age"].fillna(3)
    df["vehicle_age_factor"] = np.select(
        [vehicle_age <= 2, vehicle_age <= 5, vehicle_age <= 10],
        [1.15, 1.0, 0.9],
        default=0.85
    )
    
    # Coverage factor
    requested_coverage = inputs["requested_coverage"].fillna(100000)
    df["coverage_factor"] = (0.8 + (requested_coverage / 500000) * 0.4).clip(0.8, 1.4)
    
    # Deductible credit (higher deductible = lower premium)
    deductible = inputs["requested_deductible"].fillna(500)
    df["deductible_credit"] = (1.0 - (deductible / 5000)).clip(0.7, 1.0)
    
    # Base premium calculation
    base_rate = 800  # Base annual rate
    df["estimated_base_premium"] = (
        base_rate *
        df["age_factor"] *
        df["location_factor"] *
        df["vehicle_age_factor"] *
        df["coverage_factor"] *
        df["deductible_credit"]
    ).round(2)
    
    # Apply credit score adjustment
    credit_score = inputs["credit_score"].fillna(700)
    credit_adj = np.select(
        [credit_score >= 800, credit_score >= 700, credit_score >= 600],
        [0.85, 1.0, 1.15],
        default=1.35
    )
    
    df["estimated_annual_premium"] = (df["estimated_base_premium"] * credit_adj).round(2)
    df["estimated_monthly_premium"] = (df["estimated_annual_premium"] / 12).round(2)
    
    return df


# =============================================================================
# ON-DEMAND FEATURE VIEWS - Fraud Detection (Streaming/DSS)
# =============================================================================

@on_demand_feature_view(
    sources=[
        transaction_features_fv,
        fraud_detection_request,
    ],
    schema=[
        Field(name="velocity_risk_score", dtype=Float64),
        Field(name="amount_risk_score", dtype=Float64),
        Field(name="device_risk_score", dtype=Float64),
        Field(name="merchant_risk_score", dtype=Float64),
        Field(name="combined_fraud_score", dtype=Float64),
        Field(name="fraud_decision", dtype=String),
        Field(name="risk_factors", dtype=String),
        Field(name="recommended_action", dtype=String),
    ],
    mode="pandas",
    description="Real-time fraud detection scoring",
    tags={"domain": "streaming", "use_case": "fraud_detection", "real_time": "true"},
)
def fraud_detection_score(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate real-time fraud detection scores for transactions.
    
    Combines historical transaction patterns with current transaction
    context to generate fraud risk assessment.
    """
    df = pd.DataFrame()
    
    # Velocity risk (transaction frequency anomaly)
    txn_count_1h = inputs["txn_count_last_hour"].fillna(0)
    normal_hourly_rate = 2  # Assume normal is 2 transactions per hour
    df["velocity_risk_score"] = (
        (txn_count_1h / normal_hourly_rate) * 25
    ).clip(0, 100)
    
    # Amount risk (transaction amount anomaly)
    current_amount = inputs["transaction_amount"]
    # Simple anomaly: compare to a threshold (would normally use customer's average)
    amount_threshold = 1000  # Simplified threshold
    df["amount_risk_score"] = (
        (current_amount / amount_threshold) * 30
    ).clip(0, 100)
    
    # Device risk
    device_trust = inputs["device_trust_score"].fillna(0.5)
    device_match = inputs.get("device_fingerprint_match", True)
    df["device_risk_score"] = (
        (1 - device_trust) * 50 +
        (~device_match.fillna(True)).astype(int) * 50
    ).clip(0, 100)
    
    # Merchant risk
    high_risk = inputs["high_risk_merchant_flag"].fillna(False)
    merchant_score = inputs.get("merchant_risk_score", 0.5).fillna(0.5)
    df["merchant_risk_score"] = (
        high_risk.astype(int) * 50 +
        merchant_score * 50
    ).clip(0, 100)
    
    # Combined fraud score (weighted)
    df["combined_fraud_score"] = (
        df["velocity_risk_score"] * 0.25 +
        df["amount_risk_score"] * 0.30 +
        df["device_risk_score"] * 0.25 +
        df["merchant_risk_score"] * 0.20
    ).round(2)
    
    # Fraud decision
    df["fraud_decision"] = pd.cut(
        df["combined_fraud_score"],
        bins=[0, 30, 60, 80, 100],
        labels=["approve", "review", "challenge", "decline"],
    ).astype(str)
    
    # Risk factors (identify top contributing factors)
    risk_factors_list = []
    for idx in range(len(df)):
        factors = []
        if df.loc[idx, "velocity_risk_score"] > 50:
            factors.append("HIGH_VELOCITY")
        if df.loc[idx, "amount_risk_score"] > 50:
            factors.append("UNUSUAL_AMOUNT")
        if df.loc[idx, "device_risk_score"] > 50:
            factors.append("DEVICE_RISK")
        if df.loc[idx, "merchant_risk_score"] > 50:
            factors.append("MERCHANT_RISK")
        if inputs.loc[idx, "is_international"]:
            factors.append("INTERNATIONAL")
        risk_factors_list.append(";".join(factors) if factors else "NONE")
    df["risk_factors"] = risk_factors_list
    
    # Recommended action
    actions = {
        "approve": "PROCEED",
        "review": "3DS_CHALLENGE",
        "challenge": "STEP_UP_AUTH",
        "decline": "BLOCK_TRANSACTION",
    }
    df["recommended_action"] = df["fraud_decision"].map(actions)
    
    return df


# =============================================================================
# ON-DEMAND FEATURE VIEWS - Claims Assessment (Batch)
# =============================================================================

@on_demand_feature_view(
    sources=[
        claims_aggregation_fv,
        provider_fv,
        claims_assessment_request,
    ],
    schema=[
        Field(name="claim_validity_score", dtype=Float64),
        Field(name="fraud_risk_indicator", dtype=Float64),
        Field(name="provider_trust_score", dtype=Float64),
        Field(name="documentation_quality_score", dtype=Float64),
        Field(name="recommended_reserve", dtype=Float64),
        Field(name="expected_payout_ratio", dtype=Float64),
        Field(name="fast_track_eligible", dtype=Bool),
        Field(name="siu_referral_recommended", dtype=Bool),
        Field(name="priority_level", dtype=String),
        Field(name="estimated_processing_days", dtype=Int32),
    ],
    mode="pandas",
    description="Claims assessment and fraud indicator calculation",
    tags={"domain": "claims", "use_case": "batch"},
)
def claims_fraud_indicators(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate claims assessment scores and fraud indicators.
    
    Combines customer claims history, provider information, and current
    claim details to assess validity and fraud risk.
    """
    df = pd.DataFrame()
    
    # Claim validity score based on documentation and circumstances
    doc_score = inputs["documentation_score"].fillna(50)
    has_witnesses = inputs["has_witnesses"].fillna(False).astype(int) * 10
    police_report = inputs["police_report_filed"].fillna(False).astype(int) * 15
    days_reported = inputs["days_since_incident"].fillna(30)
    timeliness_score = (30 - days_reported.clip(0, 30)) / 30 * 25  # Report within 30 days
    
    df["claim_validity_score"] = (
        doc_score * 0.5 + has_witnesses + police_report + timeliness_score
    ).clip(0, 100)
    
    # Documentation quality score
    df["documentation_quality_score"] = (doc_score + police_report * 2 + has_witnesses).clip(0, 100)
    
    # Fraud risk indicator
    claim_amount = inputs["claim_amount_requested"]
    avg_claim = inputs["avg_claim_amount"].fillna(5000)
    amount_ratio = claim_amount / avg_claim.clip(1, None)
    
    suspicious_ratio = inputs["suspicious_claim_ratio"].fillna(0)
    fraud_history = inputs["fraud_claims_count"].fillna(0) > 0
    provider_fraud_score = inputs.get("fraud_risk_score", 0.1).fillna(0.1)
    
    df["fraud_risk_indicator"] = (
        (amount_ratio - 1).clip(0, 2) * 20 +  # Unusual amount
        suspicious_ratio * 30 +  # Historical suspicious claims
        fraud_history.astype(int) * 30 +  # Past fraud
        provider_fraud_score * 20  # Provider fraud risk
    ).clip(0, 100)
    
    # Provider trust score
    provider_quality = inputs.get("quality_score", 3.0).fillna(3.0)
    provider_years = inputs.get("years_in_network", 0).fillna(0)
    df["provider_trust_score"] = (
        provider_quality * 15 +  # Up to 75
        (provider_years / 10).clip(0, 1) * 25  # Up to 25
    ).clip(0, 100)
    
    # Recommended reserve calculation
    historical_approval_ratio = inputs["avg_approval_ratio"].fillna(0.8)
    df["recommended_reserve"] = (
        claim_amount * historical_approval_ratio * 
        (1 + df["fraud_risk_indicator"] / 200)  # Increase reserve for high fraud risk
    ).round(2)
    
    # Expected payout ratio
    df["expected_payout_ratio"] = (
        historical_approval_ratio * 
        (1 - df["fraud_risk_indicator"] / 200)
    ).clip(0, 1).round(4)
    
    # Fast track eligibility
    df["fast_track_eligible"] = (
        (df["claim_validity_score"] >= 70) &
        (df["fraud_risk_indicator"] <= 30) &
        (claim_amount <= 10000) &
        (df["documentation_quality_score"] >= 60)
    )
    
    # SIU referral recommendation
    df["siu_referral_recommended"] = (
        (df["fraud_risk_indicator"] >= 60) |
        (fraud_history) |
        (amount_ratio > 3) |
        (inputs.get("suspended_flag", False).fillna(False))
    )
    
    # Priority level
    df["priority_level"] = pd.cut(
        df["claim_validity_score"] + (100 - df["fraud_risk_indicator"]),
        bins=[0, 80, 120, 160, 200],
        labels=["low", "medium", "high", "urgent"],
    ).astype(str)
    
    # Estimated processing days
    base_days = 14
    df["estimated_processing_days"] = (
        base_days +
        (df["fraud_risk_indicator"] > 50).astype(int) * 10 +  # Extra review time
        (claim_amount > 50000).astype(int) * 7 +  # Large claims
        (~df["fast_track_eligible"]).astype(int) * 5
    ).clip(3, 60)
    
    return df
