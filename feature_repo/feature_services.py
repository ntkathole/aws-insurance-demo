"""
Feature Services for AWS Insurance Demo.

Feature Services group related features for specific use cases and model versions.
They provide:
- A stable interface for model serving
- Feature versioning and governance
- Logging configuration for monitoring
- Access control boundaries

Use Cases:
- underwriting_v1/v2: Real-time auto underwriting (PCM)
- claims_assessment_v1: Batch claims optimization
- fraud_detection_v1: Real-time fraud detection (DSS)
"""

from feast import FeatureService
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.contrib.redshift_offline_store.redshift import (
    RedshiftLoggingDestination,
)

# Import feature views
from feature_views.underwriting_features import (
    customer_profile_fv,
    customer_credit_fv,
    customer_risk_fv,
    customer_risk_fresh_fv,
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

# Import on-demand feature views
from on_demand_features import (
    underwriting_risk_score,
    premium_calculator,
    fraud_detection_score,
    claims_fraud_indicators,
)


# =============================================================================
# UNDERWRITING FEATURE SERVICES (Real-Time PCM)
# =============================================================================

# Underwriting V1 - Basic risk assessment
underwriting_v1 = FeatureService(
    name="underwriting_v1",
    features=[
        # Customer profile features (subset for basic underwriting)
        customer_profile_fv[[
            "age",
            "gender",
            "state",
            "region_risk_zone",
            "customer_tenure_months",
            "num_policies",
        ]],
        # Credit features
        customer_credit_fv[[
            "credit_score",
            "credit_score_tier",
            "insurance_score",
            "bankruptcy_flag",
        ]],
        # Risk features (subset)
        customer_risk_fv[[
            "overall_risk_score",
            "claims_risk_score",
            "num_claims_3y",
            "risk_segment",
        ]],
        # On-demand risk score calculation
        underwriting_risk_score,
    ],
    tags={
        "version": "1.0",
        "use_case": "pcm",
        "latency_sla_ms": "50",
        "owner": "underwriting-team",
    },
    description="Basic underwriting feature service for auto policy quotes",
)

# Underwriting V2 - Comprehensive risk assessment with all features
underwriting_v2 = FeatureService(
    name="underwriting_v2",
    features=[
        # Full customer profile
        customer_profile_fv,
        # Full credit features
        customer_credit_fv,
        # Full risk features (fresh version for real-time updates)
        customer_risk_fresh_fv,
        # Policy features
        policy_fv,
        # On-demand calculations
        underwriting_risk_score,
        premium_calculator,
    ],
    # Enable feature logging for monitoring
    logging_config=LoggingConfig(
        destination=RedshiftLoggingDestination(
            table_name="insurance.feature_logs_underwriting"
        ),
    ),
    tags={
        "version": "2.0",
        "use_case": "pcm",
        "latency_sla_ms": "100",
        "owner": "underwriting-team",
        "model": "underwriting_model_v2",
    },
    description="Comprehensive underwriting feature service with all risk factors",
)

# Underwriting - Quick Quote (minimal features for fast response)
underwriting_quick_quote = FeatureService(
    name="underwriting_quick_quote",
    features=[
        customer_profile_fv[[
            "age",
            "state",
            "region_risk_zone",
        ]],
        customer_credit_fv[[
            "credit_score_tier",
            "insurance_score",
        ]],
        customer_risk_fv[[
            "overall_risk_score",
            "risk_segment",
        ]],
        premium_calculator,
    ],
    tags={
        "version": "1.0",
        "use_case": "pcm",
        "latency_sla_ms": "20",
        "owner": "underwriting-team",
        "tier": "quick",
    },
    description="Quick quote feature service with minimal latency",
)


# =============================================================================
# CLAIMS FEATURE SERVICES (Batch - Claims & Labs)
# =============================================================================

# Claims Assessment V1 - Standard claims processing
claims_assessment_v1 = FeatureService(
    name="claims_assessment_v1",
    features=[
        # Claims history
        claims_history_fv,
        # Claims aggregations
        claims_aggregation_fv,
        # Provider features
        provider_fv,
        # On-demand fraud indicators
        claims_fraud_indicators,
    ],
    logging_config=LoggingConfig(
        destination=RedshiftLoggingDestination(
            table_name="insurance.feature_logs_claims"
        ),
    ),
    tags={
        "version": "1.0",
        "use_case": "claims",
        "batch": "true",
        "owner": "claims-team",
    },
    description="Standard claims assessment feature service",
)

# Claims Assessment with Lab Results - For health/life claims
claims_with_labs_v1 = FeatureService(
    name="claims_with_labs_v1",
    features=[
        claims_history_fv,
        claims_aggregation_fv,
        lab_results_fv,  # Include lab results
        provider_fv,
        claims_fraud_indicators,
    ],
    tags={
        "version": "1.0",
        "use_case": "claims",
        "batch": "true",
        "hipaa": "true",
        "owner": "claims-team",
    },
    description="Claims assessment with lab results for health claims",
)

# Claims Fraud Detection - SIU focused
claims_fraud_detection_v1 = FeatureService(
    name="claims_fraud_detection_v1",
    features=[
        claims_history_fv[[
            "claim_type",
            "claim_amount_requested",
            "claim_amount_approved",
            "injury_flag",
            "litigation_flag",
            "attorney_represented",
            "siu_referral",
            "fraud_score",
            "suspicious_indicators_count",
        ]],
        claims_aggregation_fv[[
            "total_claims_lifetime",
            "claims_count_1y",
            "claims_count_3y",
            "fraud_claims_count",
            "siu_referral_count",
            "suspicious_claim_ratio",
            "litigation_rate",
        ]],
        provider_fv[[
            "fraud_risk_score",
            "billing_anomaly_score",
            "siu_investigation_count",
            "suspended_flag",
        ]],
        claims_fraud_indicators,
    ],
    tags={
        "version": "1.0",
        "use_case": "siu",
        "batch": "true",
        "owner": "siu-team",
    },
    description="Fraud-focused feature service for SIU investigations",
)


# =============================================================================
# FRAUD DETECTION FEATURE SERVICES (Streaming/DSS)
# =============================================================================

# Transaction Fraud Detection V1 - Real-time
fraud_detection_v1 = FeatureService(
    name="fraud_detection_v1",
    features=[
        transaction_features_fv,
        fraud_detection_score,
    ],
    tags={
        "version": "1.0",
        "use_case": "dss",
        "real_time": "true",
        "latency_sla_ms": "50",
        "owner": "fraud-team",
    },
    description="Real-time transaction fraud detection feature service",
)


# =============================================================================
# COMBINED FEATURE SERVICES
# =============================================================================

# Full Customer 360 View - Combines all customer features
customer_360_v1 = FeatureService(
    name="customer_360_v1",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        claims_aggregation_fv,
        policy_fv,
    ],
    tags={
        "version": "1.0",
        "use_case": "analytics",
        "owner": "data-science-team",
    },
    description="Complete customer view for analytics and modeling",
)

# Risk Assessment Bundle - All risk-related features
risk_assessment_bundle_v1 = FeatureService(
    name="risk_assessment_bundle_v1",
    features=[
        customer_credit_fv,
        customer_risk_fv,
        claims_aggregation_fv[[
            "claims_count_3y",
            "total_claims_amount_3y",
            "avg_claim_amount",
            "fraud_claims_count",
            "suspicious_claim_ratio",
        ]],
        underwriting_risk_score,
    ],
    tags={
        "version": "1.0",
        "use_case": "risk",
        "owner": "risk-team",
    },
    description="Bundled risk features for comprehensive risk assessment",
)


# =============================================================================
# BENCHMARK/TESTING FEATURE SERVICES
# =============================================================================

# Benchmark service - For latency testing
benchmark_small = FeatureService(
    name="benchmark_small",
    features=[
        customer_profile_fv[[
            "age",
            "gender",
            "state",
        ]],
    ],
    tags={
        "use_case": "benchmark",
        "size": "small",
        "expected_features": "3",
    },
    description="Small feature set for latency benchmarking",
)

benchmark_medium = FeatureService(
    name="benchmark_medium",
    features=[
        customer_profile_fv[[
            "age",
            "gender",
            "state",
            "region_risk_zone",
            "customer_tenure_months",
            "num_policies",
        ]],
        customer_credit_fv[[
            "credit_score",
            "insurance_score",
            "debt_to_income_ratio",
        ]],
        customer_risk_fv[[
            "overall_risk_score",
            "claims_risk_score",
            "num_claims_3y",
        ]],
    ],
    tags={
        "use_case": "benchmark",
        "size": "medium",
        "expected_features": "12",
    },
    description="Medium feature set for latency benchmarking",
)

benchmark_large = FeatureService(
    name="benchmark_large",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
    ],
    tags={
        "use_case": "benchmark",
        "size": "large",
        "expected_features": "40+",
    },
    description="Large feature set for latency benchmarking",
)

benchmark_with_odfv = FeatureService(
    name="benchmark_with_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_risk_score,
        premium_calculator,
    ],
    tags={
        "use_case": "benchmark",
        "size": "large_with_odfv",
        "expected_features": "50+",
    },
    description="Large feature set with on-demand transformations for benchmarking",
)
