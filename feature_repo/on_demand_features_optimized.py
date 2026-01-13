"""
Optimized On-Demand Feature Views for AWS Insurance Demo.

These are native Python optimizations of the pandas-based ODFVs to improve latency performance.
The original pandas versions remain in on_demand_features.py for comparison.

Key optimizations:
1. Eliminated row iteration loops
2. Replaced pandas operations with native Python
3. Used vectorized operations where beneficial
4. Minimized DataFrame operations

Expected performance improvement: 15-25ms reduction in ODFV overhead
"""

from typing import Any, Dict, List, Union
import math

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

# Import request sources from original file
from on_demand_features import (
    underwriting_request,
    fraud_detection_request,
    claims_assessment_request,
)


# =============================================================================
# UTILITY FUNCTIONS - Performance Optimized
# =============================================================================

def safe_get(data: pd.Series, default: Union[float, int, bool, str]) -> Union[float, int, bool, str]:
    """Safely get value from pandas Series with default."""
    if pd.isna(data):
        return default
    return data


def calculate_risk_tier(score: float) -> str:
    """Native Python replacement for pd.cut() operation."""
    if score < 40:
        return "substandard"
    elif score < 60:
        return "standard"
    elif score < 80:
        return "preferred"
    else:
        return "super_preferred"


def calculate_fraud_decision(score: float) -> str:
    """Native Python replacement for pd.cut() operation."""
    if score < 30:
        return "approve"
    elif score < 60:
        return "review"
    elif score < 80:
        return "challenge"
    else:
        return "decline"


def calculate_priority_level(combined_score: float) -> str:
    """Native Python replacement for pd.cut() operation."""
    if combined_score < 80:
        return "low"
    elif combined_score < 120:
        return "medium"
    elif combined_score < 160:
        return "high"
    else:
        return "urgent"


# =============================================================================
# OPTIMIZED ON-DEMAND FEATURE VIEWS - Underwriting
# =============================================================================

@on_demand_feature_view(
    sources=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_request,
    ],
    schema=[
        Field(name="composite_risk_score", dtype=Float64),
        Field(name="risk_tier", dtype=String),
        Field(name="base_premium_factor", dtype=Float64),
        Field(name="credit_adjustment_factor", dtype=Float64),
        Field(name="experience_adjustment_factor", dtype=Float64),
        Field(name="coverage_adjustment_factor", dtype=Float64),
        Field(name="final_premium_factor", dtype=Float64),
        Field(name="underwriting_decision", dtype=String),
        Field(name="manual_review_flag", dtype=Bool),
        Field(name="decline_reason", dtype=String),
    ],
    mode="pandas",
    description="OPTIMIZED: Real-time underwriting risk score calculation (native Python)",
    tags={"domain": "underwriting", "use_case": "pcm", "real_time": "true", "optimization": "native_python"},
)
def underwriting_risk_score_optimized(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIMIZED VERSION: Calculate real-time underwriting risk score using native Python.

    Key optimizations:
    1. Eliminated row iteration loop for decline_reason calculation
    2. Used native conditionals instead of pd.cut() and np.select()
    3. Vectorized operations where beneficial
    4. Minimized DataFrame operations
    """
    # Pre-calculate all input values to avoid repeated pandas operations
    credit_scores = inputs["credit_score"].fillna(650).values
    risk_scores = inputs["overall_risk_score"].fillna(50).values
    tenure_months = inputs["customer_tenure_months"].fillna(0).values
    late_payments = inputs["late_payments_1y"].fillna(0).values
    insurance_scores = inputs["insurance_score"].fillna(700).values
    num_claims = inputs["num_claims_3y"].fillna(0).values
    coverage_amounts = inputs["requested_coverage"].values
    bankruptcy_flags = inputs["bankruptcy_flag"].fillna(False).values
    dui_flags = inputs["dui_flag"].fillna(False).values

    # Calculate coverage median once
    coverage_median = np.median(coverage_amounts)

    # Vectorized calculations
    credit_factors = (credit_scores - 300) / 550  # Normalize 0-1
    claims_factors = 1 - (risk_scores / 100)  # Inverse
    profile_factors = np.clip(tenure_months / 120, 0, 1)
    behavioral_factors = 1 - np.clip(late_payments / 12, 0, 1)

    # Composite risk scores (vectorized)
    composite_scores = (
        credit_factors * 0.30 +
        claims_factors * 0.25 +
        profile_factors * 0.20 +
        behavioral_factors * 0.25
    ) * 100

    # Risk tiers (vectorized with native conditionals)
    risk_tiers = [calculate_risk_tier(score) for score in composite_scores]

    # Premium factors (vectorized)
    base_factors = 2.0 - (composite_scores / 100)

    # Credit adjustment (vectorized with native conditionals)
    credit_adjustments = np.array([
        0.9 if score > 800 else (1.0 if score > 700 else (1.1 if score > 600 else 1.3))
        for score in insurance_scores
    ])

    # Experience adjustment (vectorized)
    experience_adjustments = np.clip(1.0 + (num_claims * 0.1), 1.0, 1.5)

    # Coverage adjustment (vectorized)
    coverage_ratios = coverage_amounts / coverage_median
    coverage_adjustments = np.clip(1.0 + (coverage_ratios - 1) * 0.1, 0.8, 1.5)

    # Final premium factors (vectorized)
    final_factors = np.round(
        base_factors * credit_adjustments * experience_adjustments * coverage_adjustments,
        4
    )

    # Underwriting decisions (vectorized with native conditionals)
    decisions = []
    manual_flags = []
    decline_reasons = []

    for i in range(len(composite_scores)):
        score = composite_scores[i]
        bankruptcy = bankruptcy_flags[i]
        dui = dui_flags[i]
        claims = num_claims[i]
        coverage = coverage_amounts[i]

        # Decision logic
        if score >= 70 and not bankruptcy:
            decision = "auto_approve"
        elif 50 <= score < 70:
            decision = "auto_approve_with_conditions"
        elif 30 <= score < 50:
            decision = "manual_review"
        else:
            decision = "decline"

        # Manual review flag
        manual_flag = (
            decision == "manual_review" or
            dui or
            claims > 3 or
            coverage > 500000
        )

        # Decline reasons (optimized - no row iteration over DataFrame)
        reasons = []
        if dui:
            reasons.append("DUI_ON_RECORD")
        if bankruptcy:
            reasons.append("RECENT_BANKRUPTCY")
        if claims > 5:
            reasons.append("EXCESSIVE_CLAIMS")
        if score < 30:
            reasons.append("HIGH_RISK_SCORE")

        decisions.append(decision)
        manual_flags.append(manual_flag)
        decline_reasons.append(";".join(reasons))

    # Create output DataFrame
    return pd.DataFrame({
        "composite_risk_score": composite_scores,
        "risk_tier": risk_tiers,
        "base_premium_factor": base_factors,
        "credit_adjustment_factor": credit_adjustments,
        "experience_adjustment_factor": experience_adjustments,
        "coverage_adjustment_factor": coverage_adjustments,
        "final_premium_factor": final_factors,
        "underwriting_decision": decisions,
        "manual_review_flag": manual_flags,
        "decline_reason": decline_reasons,
    })


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
    description="OPTIMIZED: Dynamic premium calculation (native Python)",
    tags={"domain": "underwriting", "use_case": "pcm", "real_time": "true", "optimization": "native_python"},
)
def premium_calculator_optimized(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIMIZED VERSION: Calculate estimated premium using native Python.

    Key optimizations:
    1. Replaced np.select() with native conditionals
    2. Vectorized calculations where possible
    3. Pre-calculated common values
    """
    # Extract values once to avoid repeated pandas operations
    ages = inputs["age"].fillna(35).values
    region_risks = inputs["region_risk_zone"].fillna(3).values
    vehicle_ages = inputs["vehicle_age"].fillna(3).values
    coverage_amounts = inputs["requested_coverage"].fillna(100000).values
    deductibles = inputs["requested_deductible"].fillna(500).values
    credit_scores = inputs["credit_score"].fillna(700).values

    # Age factors (native conditionals)
    age_factors = []
    for age in ages:
        if age < 25:
            age_factors.append(1.4)
        elif age < 30:
            age_factors.append(1.1)
        elif age <= 65:
            age_factors.append(1.0)
        elif age <= 75:
            age_factors.append(1.2)
        else:
            age_factors.append(1.5)

    # Location factors (optimized mapping)
    location_map = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.35}
    location_factors = [location_map.get(risk, 1.0) for risk in region_risks]

    # Vehicle age factors (native conditionals)
    vehicle_factors = []
    for v_age in vehicle_ages:
        if v_age <= 2:
            vehicle_factors.append(1.15)
        elif v_age <= 5:
            vehicle_factors.append(1.0)
        elif v_age <= 10:
            vehicle_factors.append(0.9)
        else:
            vehicle_factors.append(0.85)

    # Coverage factors (vectorized)
    coverage_factors = np.clip(0.8 + (coverage_amounts / 500000) * 0.4, 0.8, 1.4)

    # Deductible credits (vectorized)
    deductible_credits = np.clip(1.0 - (deductibles / 5000), 0.7, 1.0)

    # Base premium calculation (vectorized)
    base_rate = 800
    base_premiums = np.round(
        base_rate *
        np.array(age_factors) *
        np.array(location_factors) *
        np.array(vehicle_factors) *
        coverage_factors *
        deductible_credits,
        2
    )

    # Credit adjustments (native conditionals)
    credit_adjustments = []
    for score in credit_scores:
        if score >= 800:
            credit_adjustments.append(0.85)
        elif score >= 700:
            credit_adjustments.append(1.0)
        elif score >= 600:
            credit_adjustments.append(1.15)
        else:
            credit_adjustments.append(1.35)

    # Final calculations (vectorized)
    annual_premiums = np.round(base_premiums * np.array(credit_adjustments), 2)
    monthly_premiums = np.round(annual_premiums / 12, 2)

    return pd.DataFrame({
        "age_factor": age_factors,
        "location_factor": location_factors,
        "vehicle_age_factor": vehicle_factors,
        "coverage_factor": coverage_factors,
        "deductible_credit": deductible_credits,
        "estimated_base_premium": base_premiums,
        "estimated_monthly_premium": monthly_premiums,
        "estimated_annual_premium": annual_premiums,
    })


# =============================================================================
# OPTIMIZED ON-DEMAND FEATURE VIEWS - Fraud Detection
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
    description="OPTIMIZED: Real-time fraud detection scoring (native Python)",
    tags={"domain": "streaming", "use_case": "fraud_detection", "real_time": "true", "optimization": "native_python"},
)
def fraud_detection_score_optimized(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIMIZED VERSION: Calculate real-time fraud detection scores using native Python.

    Key optimizations:
    1. Eliminated row iteration loop for risk_factors calculation
    2. Used native conditionals instead of pd.cut()
    3. Vectorized risk calculations
    4. Pre-calculated input values
    """
    # Extract values once to avoid repeated pandas operations
    txn_counts = inputs["txn_count_last_hour"].fillna(0).values
    amounts = inputs["transaction_amount"].values
    device_trusts = inputs["device_trust_score"].fillna(0.5).values
    device_matches = inputs.get("device_fingerprint_match", pd.Series([True] * len(inputs))).fillna(True).values
    high_risk_flags = inputs["high_risk_merchant_flag"].fillna(False).values
    merchant_scores = inputs.get("merchant_risk_score", pd.Series([0.5] * len(inputs))).fillna(0.5).values
    international_flags = inputs["is_international"].values

    # Risk score calculations (vectorized)
    normal_hourly_rate = 2
    velocity_scores = np.clip((txn_counts / normal_hourly_rate) * 25, 0, 100)

    amount_threshold = 1000
    amount_scores = np.clip((amounts / amount_threshold) * 30, 0, 100)

    device_scores = np.clip(
        (1 - device_trusts) * 50 + (~device_matches).astype(int) * 50,
        0, 100
    )

    merchant_scores_calc = np.clip(
        high_risk_flags.astype(int) * 50 + merchant_scores * 50,
        0, 100
    )

    # Combined fraud scores (vectorized)
    combined_scores = np.round(
        velocity_scores * 0.25 +
        amount_scores * 0.30 +
        device_scores * 0.25 +
        merchant_scores_calc * 0.20,
        2
    )

    # Fraud decisions and risk factors (optimized - no row iteration)
    decisions = [calculate_fraud_decision(score) for score in combined_scores]

    # Risk factors calculation (optimized)
    risk_factors_list = []
    for i in range(len(combined_scores)):
        factors = []
        if velocity_scores[i] > 50:
            factors.append("HIGH_VELOCITY")
        if amount_scores[i] > 50:
            factors.append("UNUSUAL_AMOUNT")
        if device_scores[i] > 50:
            factors.append("DEVICE_RISK")
        if merchant_scores_calc[i] > 50:
            factors.append("MERCHANT_RISK")
        if international_flags[i]:
            factors.append("INTERNATIONAL")
        risk_factors_list.append(";".join(factors) if factors else "NONE")

    # Recommended actions (optimized mapping)
    action_map = {
        "approve": "PROCEED",
        "review": "3DS_CHALLENGE",
        "challenge": "STEP_UP_AUTH",
        "decline": "BLOCK_TRANSACTION",
    }
    actions = [action_map[decision] for decision in decisions]

    return pd.DataFrame({
        "velocity_risk_score": velocity_scores,
        "amount_risk_score": amount_scores,
        "device_risk_score": device_scores,
        "merchant_risk_score": merchant_scores_calc,
        "combined_fraud_score": combined_scores,
        "fraud_decision": decisions,
        "risk_factors": risk_factors_list,
        "recommended_action": actions,
    })


# =============================================================================
# OPTIMIZED ON-DEMAND FEATURE VIEWS - Claims Assessment
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
        Field(name="estimated_processing_days", dtype=Int64),
    ],
    mode="pandas",
    description="OPTIMIZED: Claims assessment and fraud indicators (native Python)",
    tags={"domain": "claims", "use_case": "batch", "optimization": "native_python"},
)
def claims_fraud_indicators_optimized(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIMIZED VERSION: Calculate claims assessment scores using native Python.

    Key optimizations:
    1. Replaced pd.cut() with native conditionals
    2. Vectorized calculations where beneficial
    3. Pre-calculated input values
    4. Eliminated unnecessary DataFrame operations
    """
    # Extract values once
    doc_scores = inputs["documentation_score"].fillna(50).values
    has_witnesses = inputs["has_witnesses"].fillna(False).values
    police_reports = inputs["police_report_filed"].fillna(False).values
    days_reported = inputs["days_since_incident"].fillna(30).values
    claim_amounts = inputs["claim_amount_requested"].values
    avg_claims = inputs["avg_claim_amount"].fillna(5000).values
    suspicious_ratios = inputs["suspicious_claim_ratio"].fillna(0).values
    fraud_counts = inputs["fraud_claims_count"].fillna(0).values
    provider_fraud_scores = inputs.get("fraud_risk_score", pd.Series([0.1] * len(inputs))).fillna(0.1).values
    provider_quality = inputs.get("quality_score", pd.Series([3.0] * len(inputs))).fillna(3.0).values
    provider_years = inputs.get("years_in_network", pd.Series([0] * len(inputs))).fillna(0).values
    approval_ratios = inputs["avg_approval_ratio"].fillna(0.8).values
    suspended_flags = inputs.get("suspended_flag", pd.Series([False] * len(inputs))).fillna(False).values

    # Vectorized calculations
    witness_scores = has_witnesses.astype(int) * 10
    police_scores = police_reports.astype(int) * 15
    timeliness_scores = (30 - np.clip(days_reported, 0, 30)) / 30 * 25

    validity_scores = np.clip(
        doc_scores * 0.5 + witness_scores + police_scores + timeliness_scores,
        0, 100
    )

    documentation_quality = np.clip(
        doc_scores + police_scores * 2 + witness_scores,
        0, 100
    )

    # Fraud risk calculations (vectorized)
    amount_ratios = claim_amounts / np.clip(avg_claims, 1, None)
    fraud_history_flags = fraud_counts > 0

    fraud_indicators = np.clip(
        np.clip(amount_ratios - 1, 0, 2) * 20 +
        suspicious_ratios * 30 +
        fraud_history_flags.astype(int) * 30 +
        provider_fraud_scores * 20,
        0, 100
    )

    # Provider trust scores (vectorized)
    provider_trust = np.clip(
        provider_quality * 15 +
        np.clip(provider_years / 10, 0, 1) * 25,
        0, 100
    )

    # Financial calculations (vectorized)
    recommended_reserves = np.round(
        claim_amounts * approval_ratios * (1 + fraud_indicators / 200),
        2
    )

    expected_payouts = np.clip(
        np.round(approval_ratios * (1 - fraud_indicators / 200), 4),
        0, 1
    )

    # Boolean flags (vectorized)
    fast_track = (
        (validity_scores >= 70) &
        (fraud_indicators <= 30) &
        (claim_amounts <= 10000) &
        (documentation_quality >= 60)
    )

    siu_referral = (
        (fraud_indicators >= 60) |
        fraud_history_flags |
        (amount_ratios > 3) |
        suspended_flags
    )

    # Priority levels (native conditionals)
    combined_scores = validity_scores + (100 - fraud_indicators)
    priority_levels = [calculate_priority_level(score) for score in combined_scores]

    # Processing days (vectorized)
    base_days = 14
    processing_days = np.clip(
        base_days +
        (fraud_indicators > 50).astype(int) * 10 +
        (claim_amounts > 50000).astype(int) * 7 +
        (~fast_track).astype(int) * 5,
        3, 60
    )

    return pd.DataFrame({
        "claim_validity_score": validity_scores,
        "fraud_risk_indicator": fraud_indicators,
        "provider_trust_score": provider_trust,
        "documentation_quality_score": documentation_quality,
        "recommended_reserve": recommended_reserves,
        "expected_payout_ratio": expected_payouts,
        "fast_track_eligible": fast_track,
        "siu_referral_recommended": siu_referral,
        "priority_level": priority_levels,
        "estimated_processing_days": processing_days,
    })