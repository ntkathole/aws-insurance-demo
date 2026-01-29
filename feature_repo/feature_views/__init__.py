"""
Feature Views for AWS Insurance Demo.

This package contains feature view definitions for latency testing:
- customer_features: Consolidated customer features (40+ columns)
"""

# Import all feature views for easy access
from .customer_features import (
    customer_consolidated_fv,
    FEATURE_GROUPS,
    PROFILE_FEATURES,
    CREDIT_FEATURES,
    RISK_FEATURES,
)

__all__ = [
    "customer_consolidated_fv",
    "FEATURE_GROUPS",
    "PROFILE_FEATURES",
    "CREDIT_FEATURES",
    "RISK_FEATURES",
]
