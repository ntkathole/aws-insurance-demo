"""
Entity definitions for the AWS Insurance Demo.

Entities represent the primary keys used to fetch features. In insurance domain:
- customer: The policyholder or applicant
- policy: An insurance policy
- claim: An insurance claim
- provider: Healthcare/service provider (for claims processing)
"""

from feast import Entity, ValueType

# =============================================================================
# CUSTOMER ENTITY
# =============================================================================
# Primary entity for underwriting and customer-level features
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.STRING,
    description="Insurance customer or policy applicant",
    tags={
        "domain": "underwriting",
        "pii": "true",
    },
)

# =============================================================================
# POLICY ENTITY
# =============================================================================
# Entity for policy-level features
policy = Entity(
    name="policy",
    join_keys=["policy_id"],
    value_type=ValueType.STRING,
    description="Insurance policy",
    tags={
        "domain": "underwriting",
    },
)

# =============================================================================
# CLAIM ENTITY
# =============================================================================
# Entity for claims processing and optimization
claim = Entity(
    name="claim",
    join_keys=["claim_id"],
    value_type=ValueType.STRING,
    description="Insurance claim",
    tags={
        "domain": "claims",
    },
)

# =============================================================================
# PROVIDER ENTITY
# =============================================================================
# Entity for provider network analysis in claims
provider = Entity(
    name="provider",
    join_keys=["provider_id"],
    value_type=ValueType.STRING,
    description="Healthcare or service provider",
    tags={
        "domain": "claims",
    },
)

# =============================================================================
# TRANSACTION ENTITY
# =============================================================================
# Entity for transaction-level features (streaming use case)
transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="Financial transaction for fraud detection",
    tags={
        "domain": "streaming",
        "use_case": "fraud_detection",
    },
)
