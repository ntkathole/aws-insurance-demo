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

# Try to import logging components - optional, may not be available in all Feast versions
try:
    from feast.feature_logging import LoggingConfig
    from feast.infra.offline_stores.redshift_source import RedshiftLoggingDestination
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    LoggingConfig = None
    RedshiftLoggingDestination = None

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

# Import optimized on-demand feature views (native Python versions)
from on_demand_features_optimized import (
    underwriting_risk_score_optimized,
    premium_calculator_optimized,
    fraud_detection_score_optimized,
    claims_fraud_indicators_optimized,
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
    # Note: logging_config can be added when RedshiftLoggingDestination is available
    # logging_config=LoggingConfig(
    #     destination=RedshiftLoggingDestination(table_name="insurance.feature_logs_underwriting"),
    # ),
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
    # Note: logging_config can be added when RedshiftLoggingDestination is available
    # logging_config=LoggingConfig(
    #     destination=RedshiftLoggingDestination(table_name="insurance.feature_logs_claims"),
    # ),
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

# =============================================================================
# ENHANCED BENCHMARK SERVICES - For Multi-Dimensional Performance Testing
# =============================================================================

# Single FV feature count scaling tests
benchmark_profile_3 = FeatureService(
    name="benchmark_profile_3",
    features=[
        customer_profile_fv[[
            "age",
            "gender",
            "state",
        ]],
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_count",
        "size": "3_features",
        "fvs": "1",
        "expected_features": "3",
    },
    description="3 features from customer profile for feature count scaling",
)

benchmark_profile_6 = FeatureService(
    name="benchmark_profile_6",
    features=[
        customer_profile_fv[[
            "age",
            "gender",
            "state",
            "marital_status",
            "occupation",
            "education_level",
        ]],
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_count",
        "size": "6_features",
        "fvs": "1",
        "expected_features": "6",
    },
    description="6 features from customer profile for feature count scaling",
)

benchmark_profile_all = FeatureService(
    name="benchmark_profile_all",
    features=[
        customer_profile_fv,  # All 13 features
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_count",
        "size": "13_features",
        "fvs": "1",
        "expected_features": "13",
    },
    description="All customer profile features for feature count scaling",
)

# Multi-FV scaling tests
benchmark_2fv = FeatureService(
    name="benchmark_2fv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_view",
        "size": "24_features",
        "fvs": "2",
        "expected_features": "24",
    },
    description="2 feature views for FV scaling tests",
)

benchmark_3fv = FeatureService(
    name="benchmark_3fv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_view",
        "size": "40_features",
        "fvs": "3",
        "expected_features": "40",
    },
    description="3 feature views for FV scaling tests",
)

benchmark_4fv = FeatureService(
    name="benchmark_4fv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        policy_fv,
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "feature_view",
        "size": "65_features",
        "fvs": "4",
        "expected_features": "65",
    },
    description="4 feature views for FV scaling tests",
)

# ODFV isolation tests
benchmark_no_odfv = FeatureService(
    name="benchmark_no_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "odfv_count",
        "odfvs": "0",
        "baseline": "true",
        "expected_features": "40",
    },
    description="Baseline service with no ODFVs for ODFV overhead measurement",
)

benchmark_1_light_odfv = FeatureService(
    name="benchmark_1_light_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        premium_calculator,  # Lightweight ODFV
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "odfv_count",
        "odfvs": "1",
        "complexity": "light",
        "expected_features": "32",
    },
    description="Service with 1 lightweight ODFV (premium_calculator)",
)

benchmark_1_heavy_odfv = FeatureService(
    name="benchmark_1_heavy_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_risk_score,  # Heavy ODFV with row iteration
    ],
    tags={
        "use_case": "benchmark",
        "dimension": "odfv_count",
        "odfvs": "1",
        "complexity": "heavy",
        "expected_features": "50",
    },
    description="Service with 1 heavy ODFV (underwriting_risk_score)",
)


# =============================================================================
# OPTIMIZED FEATURE SERVICES - Native Python ODFV Performance Comparison
# =============================================================================

# Optimized Underwriting V1 - Native Python ODFV
underwriting_v1_optimized = FeatureService(
    name="underwriting_v1_optimized",
    features=[
        # Same base features as v1
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
            "credit_score_tier",
            "insurance_score",
            "bankruptcy_flag",
        ]],
        customer_risk_fv[[
            "overall_risk_score",
            "claims_risk_score",
            "num_claims_3y",
            "risk_segment",
        ]],
        # OPTIMIZED: Native Python ODFV
        underwriting_risk_score_optimized,
    ],
    tags={
        "version": "1.0_optimized",
        "use_case": "pcm",
        "latency_sla_ms": "35",  # Expected 15-25ms improvement
        "owner": "underwriting-team",
        "optimization": "native_python",
    },
    description="OPTIMIZED: Basic underwriting with native Python ODFV for performance comparison",
)

# Optimized Underwriting V2 - Native Python ODFVs
underwriting_v2_optimized = FeatureService(
    name="underwriting_v2_optimized",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fresh_fv,
        policy_fv,
        # OPTIMIZED: Both ODFVs use native Python
        underwriting_risk_score_optimized,
        premium_calculator_optimized,
    ],
    tags={
        "version": "2.0_optimized",
        "use_case": "pcm",
        "latency_sla_ms": "75",  # Expected 30-50ms improvement
        "owner": "underwriting-team",
        "model": "underwriting_model_v2",
        "optimization": "native_python",
    },
    description="OPTIMIZED: Comprehensive underwriting with native Python ODFVs",
)

# Optimized Quick Quote - Native Python ODFV
underwriting_quick_quote_optimized = FeatureService(
    name="underwriting_quick_quote_optimized",
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
        # OPTIMIZED: Native Python premium calculator
        premium_calculator_optimized,
    ],
    tags={
        "version": "1.0_optimized",
        "use_case": "pcm",
        "latency_sla_ms": "10",  # Expected significant improvement for quick quotes
        "owner": "underwriting-team",
        "tier": "quick",
        "optimization": "native_python",
    },
    description="OPTIMIZED: Quick quote with native Python ODFV for minimal latency",
)

# Optimized Claims Assessment - Native Python ODFV
claims_assessment_v1_optimized = FeatureService(
    name="claims_assessment_v1_optimized",
    features=[
        claims_history_fv,
        claims_aggregation_fv,
        provider_fv,
        # OPTIMIZED: Native Python claims fraud indicators
        claims_fraud_indicators_optimized,
    ],
    tags={
        "version": "1.0_optimized",
        "use_case": "claims",
        "batch": "true",
        "owner": "claims-team",
        "optimization": "native_python",
    },
    description="OPTIMIZED: Claims assessment with native Python ODFV",
)

# Optimized Fraud Detection - Native Python ODFV
fraud_detection_v1_optimized = FeatureService(
    name="fraud_detection_v1_optimized",
    features=[
        transaction_features_fv,
        # OPTIMIZED: Native Python fraud detection
        fraud_detection_score_optimized,
    ],
    tags={
        "version": "1.0_optimized",
        "use_case": "dss",
        "real_time": "true",
        "latency_sla_ms": "35",  # Expected 15ms improvement
        "owner": "fraud-team",
        "optimization": "native_python",
    },
    description="OPTIMIZED: Real-time fraud detection with native Python ODFV",
)

# =============================================================================
# OPTIMIZED BENCHMARK SERVICES - For Performance Comparison Testing
# =============================================================================

# Pure FV baseline (unchanged - for comparison)
benchmark_pure_fv_baseline = FeatureService(
    name="benchmark_pure_fv_baseline",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
    ],
    tags={
        "use_case": "benchmark",
        "type": "pure_fv",
        "optimization": "baseline",
        "expected_features": "40",
    },
    description="BASELINE: Pure FV service for ODFV performance comparison",
)

# Pandas ODFV version (for comparison)
benchmark_pandas_light_odfv = FeatureService(
    name="benchmark_pandas_light_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        premium_calculator,  # Pandas version
    ],
    tags={
        "use_case": "benchmark",
        "type": "odfv_pandas",
        "complexity": "light",
        "optimization": "pandas",
        "expected_features": "32",
    },
    description="PANDAS: Light ODFV with pandas operations for comparison",
)

# Optimized light ODFV (native Python)
benchmark_optimized_light_odfv = FeatureService(
    name="benchmark_optimized_light_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        premium_calculator_optimized,  # Native Python version
    ],
    tags={
        "use_case": "benchmark",
        "type": "odfv_optimized",
        "complexity": "light",
        "optimization": "native_python",
        "expected_features": "32",
    },
    description="OPTIMIZED: Light ODFV with native Python for performance comparison",
)

# Pandas heavy ODFV version (for comparison)
benchmark_pandas_heavy_odfv = FeatureService(
    name="benchmark_pandas_heavy_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_risk_score,  # Pandas version with row iteration
    ],
    tags={
        "use_case": "benchmark",
        "type": "odfv_pandas",
        "complexity": "heavy",
        "optimization": "pandas",
        "expected_features": "50",
    },
    description="PANDAS: Heavy ODFV with pandas operations for comparison",
)

# Optimized heavy ODFV (native Python)
benchmark_optimized_heavy_odfv = FeatureService(
    name="benchmark_optimized_heavy_odfv",
    features=[
        customer_profile_fv,
        customer_credit_fv,
        customer_risk_fv,
        underwriting_risk_score_optimized,  # Native Python version
    ],
    tags={
        "use_case": "benchmark",
        "type": "odfv_optimized",
        "complexity": "heavy",
        "optimization": "native_python",
        "expected_features": "50",
    },
    description="OPTIMIZED: Heavy ODFV with native Python for performance comparison",
)
