"""
Claims Feature Views - Batch Claims Optimization

These feature views support batch claims processing, fraud detection, and 
claims optimization workflows. They are designed for larger batch operations
and can tolerate slightly higher latency.

Transaction Type: Batch (Claims & Labs)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.aggregation import Aggregation
from feast.types import Float32, Float64, Int32, Int64, String, Bool

# Import entities and data sources
import sys
sys.path.insert(0, '..')
from entities import customer, claim, provider
from data_sources import (
    claims_history_source,
    claims_aggregation_source,
    lab_results_source,
    provider_source,
)


# =============================================================================
# CLAIMS HISTORY FEATURES
# =============================================================================
# Historical claims data for pattern analysis
claims_history_fv = FeatureView(
    name="claims_history_features",
    entities=[claim],
    ttl=timedelta(days=7),  # Weekly refresh acceptable for batch
    schema=[
        # Claim identification
        Field(name="customer_id", dtype=String, description="Customer ID"),
        Field(name="policy_id", dtype=String, description="Policy ID"),
        Field(name="provider_id", dtype=String, description="Provider ID"),
        
        # Claim details
        Field(name="claim_type", dtype=String, description="Type of claim"),
        Field(name="claim_subtype", dtype=String, description="Claim subtype/category"),
        Field(name="claim_status", dtype=String, description="Current claim status"),
        Field(name="claim_amount_requested", dtype=Float64, description="Amount requested"),
        Field(name="claim_amount_approved", dtype=Float64, description="Amount approved"),
        Field(name="claim_amount_paid", dtype=Float64, description="Amount paid"),
        
        # Claim timing
        Field(name="days_to_first_contact", dtype=Int32, description="Days to first contact"),
        Field(name="days_to_settlement", dtype=Int32, description="Days to settlement"),
        Field(name="claim_age_days", dtype=Int32, description="Age of claim in days"),
        
        # Claim characteristics
        Field(name="injury_flag", dtype=Bool, description="Involves injury"),
        Field(name="fatality_flag", dtype=Bool, description="Involves fatality"),
        Field(name="litigation_flag", dtype=Bool, description="In litigation"),
        Field(name="attorney_represented", dtype=Bool, description="Claimant has attorney"),
        Field(name="num_parties_involved", dtype=Int32, description="Number of parties"),
        
        # Investigation indicators
        Field(name="siu_referral", dtype=Bool, description="Referred to SIU"),
        Field(name="fraud_score", dtype=Float32, description="Fraud probability score"),
        Field(name="suspicious_indicators_count", dtype=Int32, description="Count of suspicious indicators"),
        
        # Adjuster info
        Field(name="adjuster_id", dtype=String, description="Assigned adjuster"),
        Field(name="adjuster_workload", dtype=Int32, description="Adjuster current workload"),
        Field(name="priority_score", dtype=Float32, description="Claim priority score"),
    ],
    online=True,
    source=claims_history_source,
    tags={
        "domain": "claims",
        "use_case": "batch",
        "latency_requirement": "medium",
    },
    description="Historical claims data for claims optimization",
)


# =============================================================================
# CLAIMS AGGREGATION FEATURES
# =============================================================================
# Pre-computed aggregations at customer level for efficient retrieval
claims_aggregation_fv = FeatureView(
    name="claims_aggregation_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        # Claim counts
        Field(name="total_claims_lifetime", dtype=Int32, description="Total lifetime claims"),
        Field(name="claims_count_1y", dtype=Int32, description="Claims in last year"),
        Field(name="claims_count_3y", dtype=Int32, description="Claims in last 3 years"),
        Field(name="claims_count_5y", dtype=Int32, description="Claims in last 5 years"),
        
        # Claim amounts
        Field(name="total_claims_amount_lifetime", dtype=Float64, description="Total lifetime claims amount"),
        Field(name="total_claims_amount_1y", dtype=Float64, description="Claims amount in last year"),
        Field(name="total_claims_amount_3y", dtype=Float64, description="Claims amount in last 3 years"),
        Field(name="avg_claim_amount", dtype=Float64, description="Average claim amount"),
        Field(name="max_claim_amount", dtype=Float64, description="Maximum claim amount"),
        
        # Claim patterns
        Field(name="avg_days_between_claims", dtype=Float32, description="Avg days between claims"),
        Field(name="days_since_last_claim", dtype=Int32, description="Days since last claim"),
        Field(name="claim_frequency_score", dtype=Float32, description="Claim frequency score"),
        
        # Claim outcomes
        Field(name="pct_claims_approved", dtype=Float32, description="Percent claims approved"),
        Field(name="pct_claims_denied", dtype=Float32, description="Percent claims denied"),
        Field(name="avg_approval_ratio", dtype=Float32, description="Avg approval to requested ratio"),
        
        # Fraud indicators
        Field(name="fraud_claims_count", dtype=Int32, description="Confirmed fraud claims"),
        Field(name="siu_referral_count", dtype=Int32, description="SIU referral count"),
        Field(name="suspicious_claim_ratio", dtype=Float32, description="Ratio of suspicious claims"),
        
        # Claim types distribution
        Field(name="auto_claims_count", dtype=Int32, description="Auto claim count"),
        Field(name="property_claims_count", dtype=Int32, description="Property claim count"),
        Field(name="liability_claims_count", dtype=Int32, description="Liability claim count"),
        Field(name="medical_claims_count", dtype=Int32, description="Medical claim count"),
        
        # Settlement patterns
        Field(name="avg_settlement_days", dtype=Float32, description="Average days to settle"),
        Field(name="litigation_rate", dtype=Float32, description="Litigation rate"),
    ],
    online=True,
    source=claims_aggregation_source,
    # Define aggregations for windowed calculations
    # Note: These are registered for documentation; actual aggregation happens in source
    tags={
        "domain": "claims",
        "use_case": "batch",
        "aggregation": "true",
    },
    description="Aggregated claims metrics per customer for risk assessment",
)


# =============================================================================
# LAB RESULTS FEATURES
# =============================================================================
# Medical lab results for health/life insurance claims
lab_results_fv = FeatureView(
    name="lab_results_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        # Test identifiers
        Field(name="latest_test_date", dtype=String, description="Date of latest test"),
        Field(name="test_provider", dtype=String, description="Testing lab/provider"),
        
        # Health indicators (normalized scores 0-100)
        Field(name="overall_health_score", dtype=Float32, description="Overall health score"),
        Field(name="cardiovascular_score", dtype=Float32, description="Cardiovascular health score"),
        Field(name="metabolic_score", dtype=Float32, description="Metabolic health score"),
        Field(name="liver_function_score", dtype=Float32, description="Liver function score"),
        Field(name="kidney_function_score", dtype=Float32, description="Kidney function score"),
        
        # Risk indicators
        Field(name="bmi_category", dtype=String, description="BMI category"),
        Field(name="smoker_status", dtype=String, description="Smoker status"),
        Field(name="blood_pressure_category", dtype=String, description="BP category"),
        Field(name="cholesterol_category", dtype=String, description="Cholesterol category"),
        Field(name="diabetes_risk_level", dtype=String, description="Diabetes risk level"),
        
        # Lab values (discretized/categorized for privacy)
        Field(name="glucose_category", dtype=String, description="Glucose level category"),
        Field(name="a1c_category", dtype=String, description="A1C level category"),
        Field(name="ldl_category", dtype=String, description="LDL cholesterol category"),
        Field(name="hdl_category", dtype=String, description="HDL cholesterol category"),
        Field(name="triglycerides_category", dtype=String, description="Triglycerides category"),
        
        # Health trends
        Field(name="health_trend_6m", dtype=String, description="Health trend (improving/stable/declining)"),
        Field(name="risk_trend_6m", dtype=String, description="Risk trend"),
        Field(name="num_abnormal_results", dtype=Int32, description="Number of abnormal results"),
        
        # Compliance
        Field(name="regular_checkups_flag", dtype=Bool, description="Has regular checkups"),
        Field(name="medication_adherence_score", dtype=Float32, description="Medication adherence"),
    ],
    online=True,
    source=lab_results_source,
    tags={
        "domain": "claims",
        "use_case": "batch",
        "pii": "true",
        "hipaa": "true",
    },
    description="Medical lab results for health assessment (anonymized/categorized)",
)


# =============================================================================
# PROVIDER FEATURES
# =============================================================================
# Healthcare provider network features for claims analysis
provider_fv = FeatureView(
    name="provider_features",
    entities=[provider],
    ttl=timedelta(days=1),
    schema=[
        # Provider identification
        Field(name="provider_name", dtype=String, description="Provider name"),
        Field(name="provider_type", dtype=String, description="Provider type"),
        Field(name="specialty", dtype=String, description="Medical specialty"),
        Field(name="network_status", dtype=String, description="In-network/Out-of-network"),
        
        # Provider metrics
        Field(name="quality_score", dtype=Float32, description="Quality score (1-5)"),
        Field(name="patient_satisfaction_score", dtype=Float32, description="Patient satisfaction"),
        Field(name="cost_efficiency_score", dtype=Float32, description="Cost efficiency score"),
        Field(name="outcome_score", dtype=Float32, description="Treatment outcome score"),
        
        # Volume metrics
        Field(name="claims_volume_monthly", dtype=Int32, description="Monthly claims volume"),
        Field(name="unique_patients_monthly", dtype=Int32, description="Unique patients monthly"),
        Field(name="avg_claim_amount", dtype=Float64, description="Average claim amount"),
        
        # Fraud indicators
        Field(name="fraud_risk_score", dtype=Float32, description="Provider fraud risk score"),
        Field(name="billing_anomaly_score", dtype=Float32, description="Billing anomaly score"),
        Field(name="siu_investigation_count", dtype=Int32, description="SIU investigation count"),
        Field(name="suspended_flag", dtype=Bool, description="Currently suspended"),
        
        # Network analysis
        Field(name="referral_count_to", dtype=Int32, description="Referrals to this provider"),
        Field(name="referral_count_from", dtype=Int32, description="Referrals from this provider"),
        Field(name="network_centrality_score", dtype=Float32, description="Network centrality"),
        
        # Geographic
        Field(name="state", dtype=String, description="Provider state"),
        Field(name="region", dtype=String, description="Provider region"),
        Field(name="urban_rural", dtype=String, description="Urban/Rural"),
        
        # Accreditation
        Field(name="accredited_flag", dtype=Bool, description="Is accredited"),
        Field(name="years_in_network", dtype=Int32, description="Years in network"),
    ],
    online=True,
    source=provider_source,
    tags={
        "domain": "claims",
        "use_case": "batch",
    },
    description="Provider network features for claims analysis and fraud detection",
)
