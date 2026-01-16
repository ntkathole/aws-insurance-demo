"""
Entity definitions for the AWS Insurance Demo - Simplified Architecture.

This simplified architecture focuses on customer-centric features for latency testing.
Only the customer entity is retained to support:
- 1 batch feature view (customer_consolidated_features)
- 1 on-demand feature view (premium_calculator_optimized)

Original entities (policy, claim, provider, transaction) have been removed
as part of the architecture simplification.
"""

from feast import Entity, ValueType

# =============================================================================
# CUSTOMER ENTITY - Primary Entity for Simplified Architecture
# =============================================================================
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.STRING,
    description="Insurance customer or policy applicant - primary entity for simplified architecture",
    tags={
        "domain": "customer",
        "pii": "true",
        "simplified_architecture": "true",
        "use_case": "latency_testing",
    },
)
