"""
Feature Views for AWS Insurance Demo.

This package contains feature view definitions organized by use case:
- underwriting_features: Real-time auto underwriting (PCM)
- claims_features: Batch claims optimization
- streaming_features: Future DSS streaming features
"""

# Import all feature views for easy access
from .underwriting_features import (
    customer_profile_fv,
    customer_credit_fv,
    customer_risk_fv,
    customer_risk_fresh_fv,
    policy_fv,
)

from .claims_features import (
    claims_history_fv,
    claims_aggregation_fv,
    lab_results_fv,
    provider_fv,
)

from .streaming_features import (
    transaction_features_fv,
    # Uncomment when Kafka is configured:
    # transaction_stream_fv,
)

__all__ = [
    # Underwriting (Real-Time PCM)
    "customer_profile_fv",
    "customer_credit_fv",
    "customer_risk_fv",
    "customer_risk_fresh_fv",
    "policy_fv",
    # Claims (Batch)
    "claims_history_fv",
    "claims_aggregation_fv",
    "lab_results_fv",
    "provider_fv",
    # Streaming (DSS)
    "transaction_features_fv",
]
