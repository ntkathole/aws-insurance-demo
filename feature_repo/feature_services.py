"""
Feature Services for Simplified AWS Insurance Demo Architecture.

This simplified architecture provides feature services for latency testing with:
- 1 batch feature view (customer_consolidated_features) with variable feature counts
- 1 on-demand feature view (premium_calculator_optimized)

The services are designed to enable comprehensive latency testing by:
1. Variable feature count testing (5, 10, 20, 40 features)
2. ODFV overhead measurement (with/without premium calculator)
3. Performance comparison and bottleneck identification
"""

from feast import FeatureService

# Import simplified feature components
from feature_views.customer_features import customer_consolidated_fv, FEATURE_GROUPS
from on_demand_features_simplified import premium_calculator_optimized


# =============================================================================
# VARIABLE FEATURE COUNT SERVICES - For Latency Testing
# =============================================================================

# 5 features - Basic demographic and risk info
customer_5_features = FeatureService(
    name="customer_5_features",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["basic_5"]],
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "5",
        "type": "batch_only",
        "description": "Basic customer info for fastest retrieval",
    },
    description="5 essential customer features for baseline latency testing",
)

# 10 features - Standard underwriting set
customer_10_features = FeatureService(
    name="customer_10_features",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["standard_10"]],
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "10",
        "type": "batch_only",
        "description": "Standard underwriting features",
    },
    description="10 key customer features for standard risk assessment",
)

# 20 features - Comprehensive risk analysis
customer_20_features = FeatureService(
    name="customer_20_features",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["comprehensive_20"]],
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "20",
        "type": "batch_only",
        "description": "Comprehensive risk factors",
    },
    description="20 comprehensive customer features for detailed analysis",
)

# 40 features - Full customer profile
customer_40_features = FeatureService(
    name="customer_40_features",
    features=[
        customer_consolidated_fv,  # All features
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "40",
        "type": "batch_only",
        "description": "Complete customer profile",
    },
    description="All 40 customer features for maximum latency testing",
)


# =============================================================================
# ODFV COMBINATION SERVICES - Feature + Transformation Testing
# =============================================================================

# 5 features + Premium ODFV
customer_5_with_premium = FeatureService(
    name="customer_5_with_premium",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["basic_5"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "13",  # 5 + 8 ODFV outputs
        "type": "batch_plus_odfv",
        "odfv_count": "1",
        "description": "Minimal features + premium calculation",
    },
    description="5 customer features + premium calculator for lightweight serving",
)

# 10 features + Premium ODFV
customer_10_with_premium = FeatureService(
    name="customer_10_with_premium",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["standard_10"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "18",  # 10 + 8 ODFV outputs
        "type": "batch_plus_odfv",
        "odfv_count": "1",
        "description": "Standard features + premium calculation",
    },
    description="10 customer features + premium calculator for standard underwriting",
)

# 20 features + Premium ODFV
customer_20_with_premium = FeatureService(
    name="customer_20_with_premium",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["comprehensive_20"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "28",  # 20 + 8 ODFV outputs
        "type": "batch_plus_odfv",
        "odfv_count": "1",
        "description": "Comprehensive features + premium calculation",
    },
    description="20 customer features + premium calculator for comprehensive quotes",
)

# 40 features + Premium ODFV - Full stack
customer_40_with_premium = FeatureService(
    name="customer_40_with_premium",
    features=[
        customer_consolidated_fv,  # All 40 features
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "latency_testing",
        "feature_count": "48",  # 40 + 8 ODFV outputs
        "type": "batch_plus_odfv",
        "odfv_count": "1",
        "description": "Complete customer profile + premium calculation",
    },
    description="All customer features + premium calculator for maximum latency testing",
)


# =============================================================================
# PRODUCTION SERVICES - Real Use Cases
# =============================================================================

# Quick quote service - Optimized for speed
quick_quote = FeatureService(
    name="quick_quote",
    features=[
        customer_consolidated_fv[["age", "state", "region_risk_zone", "credit_score", "overall_risk_score"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "production",
        "tier": "quick",
        "latency_sla_ms": "50",
        "feature_count": "13",
        "description": "Fast quote with minimal features",
    },
    description="Quick premium quotes with minimal latency",
)

# Standard quote service - Balanced features
standard_quote = FeatureService(
    name="standard_quote",
    features=[
        customer_consolidated_fv[FEATURE_GROUPS["standard_10"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "production",
        "tier": "standard",
        "latency_sla_ms": "100",
        "feature_count": "18",
        "description": "Standard quote with key risk factors",
    },
    description="Standard premium quotes with key customer risk factors",
)

# Comprehensive quote service - Full analysis
comprehensive_quote = FeatureService(
    name="comprehensive_quote",
    features=[
        customer_consolidated_fv,  # All 40 features
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "production",
        "tier": "comprehensive",
        "latency_sla_ms": "200",
        "feature_count": "48",
        "description": "Comprehensive quote with full customer profile",
    },
    description="Comprehensive premium quotes with complete customer analysis",
)


# =============================================================================
# BASELINE SERVICES - For Performance Comparison
# =============================================================================

# Pure batch baseline - No ODFV for comparison
customer_baseline = FeatureService(
    name="customer_baseline",
    features=[
        customer_consolidated_fv,  # All 40 features, no ODFV
    ],
    tags={
        "use_case": "baseline",
        "type": "batch_only",
        "feature_count": "40",
        "odfv_count": "0",
        "description": "Pure batch FV serving for ODFV overhead measurement",
    },
    description="Baseline service with no ODFVs to measure pure batch FV latency",
)

# ODFV-only baseline - Just premium calculator
premium_only = FeatureService(
    name="premium_only",
    features=[
        # Only the essential features needed for premium calculator
        customer_consolidated_fv[["age", "region_risk_zone", "credit_score"]],
        premium_calculator_optimized,
    ],
    tags={
        "use_case": "baseline",
        "type": "minimal_odfv",
        "feature_count": "11",  # 3 + 8 ODFV outputs
        "odfv_count": "1",
        "description": "Minimal features for ODFV overhead isolation",
    },
    description="Minimal feature set to isolate ODFV transformation latency",
)


# =============================================================================
# FEATURE SERVICE SUMMARY
# =============================================================================
"""
Simplified Architecture Feature Services Summary:

LATENCY TESTING SERVICES:
1. customer_5_features -> customer_40_features: Variable batch FV latency (5-40 features)
2. customer_5_with_premium -> customer_40_with_premium: Variable + ODFV latency
3. customer_baseline: Pure batch FV (40 features, no ODFV)
4. premium_only: Minimal ODFV test (3 + 8 features)

PRODUCTION SERVICES:
1. quick_quote: 13 features, <50ms SLA
2. standard_quote: 18 features, <100ms SLA
3. comprehensive_quote: 48 features, <200ms SLA

TESTING CAPABILITY:
- Feature count scaling: 5 -> 10 -> 20 -> 40 features
- ODFV overhead: Compare with/without premium calculator
- Bottleneck identification: Batch FV vs ODFV transformation
- Performance regression: Compare against baselines

EXPECTED LATENCIES (targets):
- customer_5_features: ~10-15ms (batch only)
- customer_5_with_premium: ~20-30ms (batch + ODFV)
- customer_40_features: ~30-50ms (large batch)
- customer_40_with_premium: ~40-60ms (large batch + ODFV)
"""