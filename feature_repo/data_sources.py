"""
Data source definitions for the AWS Insurance Demo.

Data sources connect Feast to the underlying data storage:
- RedshiftSource for batch/offline data (Redshift tables)
- PushSource for real-time data ingestion
- KafkaSource for streaming data (future DSS)
- FileSource for local development/testing
"""

from datetime import timedelta

from feast import FileSource, PushSource, RedshiftSource
from feast.data_format import JsonFormat

# Try to import KafkaSource - it's optional and may not be available
try:
    from feast import KafkaSource
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


# =============================================================================
# REDSHIFT DATA SOURCES - Production (Offline Store)
# =============================================================================

# Customer profile data - Demographics, account history
customer_profile_source = RedshiftSource(
    name="customer_profile_source",
    table="customer_profiles",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer demographic and profile information",
    tags={"domain": "underwriting", "refresh": "daily"},
)

# Customer credit and financial data
customer_credit_source = RedshiftSource(
    name="customer_credit_source",
    table="customer_credit_data",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer credit scores and financial indicators",
    tags={"domain": "underwriting", "refresh": "hourly"},
)

# Customer risk metrics
customer_risk_source = RedshiftSource(
    name="customer_risk_source",
    table="customer_risk_metrics",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer risk assessment metrics",
    tags={"domain": "underwriting", "refresh": "hourly"},
)

# Policy information
policy_source = RedshiftSource(
    name="policy_source",
    table="policy_details",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Policy details and coverage information",
    tags={"domain": "underwriting", "refresh": "daily"},
)

# Claims history data
claims_history_source = RedshiftSource(
    name="claims_history_source",
    table="claims_history",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Historical claims data",
    tags={"domain": "claims", "refresh": "daily"},
)

# Claims aggregations (pre-computed)
claims_aggregation_source = RedshiftSource(
    name="claims_aggregation_source",
    table="claims_aggregations",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Aggregated claims metrics by customer",
    tags={"domain": "claims", "refresh": "daily"},
)

# Lab results data
lab_results_source = RedshiftSource(
    name="lab_results_source",
    table="lab_results",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Medical lab test results",
    tags={"domain": "claims", "refresh": "daily", "pii": "true"},
)

# Provider network data
provider_source = RedshiftSource(
    name="provider_source",
    table="provider_network",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Healthcare provider network information",
    tags={"domain": "claims", "refresh": "weekly"},
)

# Transaction data for fraud detection
transaction_source = RedshiftSource(
    name="transaction_source",
    table="transactions",
    schema="insurance",
    database="feast_db",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Financial transactions for fraud detection",
    tags={"domain": "streaming", "refresh": "real-time"},
)


# =============================================================================
# FILE SOURCES - Local Development/Testing
# =============================================================================

# For local development without Redshift
customer_profile_file_source = FileSource(
    name="customer_profile_file_source",
    path="data/sample/customer_profiles.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer profiles (local file for testing)",
)

customer_credit_file_source = FileSource(
    name="customer_credit_file_source",
    path="data/sample/customer_credit.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer credit data (local file for testing)",
)

customer_risk_file_source = FileSource(
    name="customer_risk_file_source",
    path="data/sample/customer_risk.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Customer risk metrics (local file for testing)",
)

claims_history_file_source = FileSource(
    name="claims_history_file_source",
    path="data/sample/claims_history.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Claims history (local file for testing)",
)

transaction_file_source = FileSource(
    name="transaction_file_source",
    path="data/sample/transactions.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Transactions (local file for testing)",
)


# =============================================================================
# PUSH SOURCES - Real-Time Data Ingestion
# =============================================================================

# Push source for real-time customer risk updates
customer_risk_push_source = PushSource(
    name="customer_risk_push_source",
    batch_source=customer_risk_source,
    description="Push source for real-time customer risk updates",
    tags={"domain": "underwriting", "real_time": "true"},
)

# Push source for real-time transaction data
transaction_push_source = PushSource(
    name="transaction_push_source",
    batch_source=transaction_source,
    description="Push source for real-time transaction data",
    tags={"domain": "streaming", "real_time": "true"},
)


# =============================================================================
# KAFKA SOURCES - Streaming (Future DSS)
# =============================================================================

# Note: KafkaSource is optional and requires 'feast[kafka]' installation
# These sources are prepared for future streaming use cases

if KAFKA_AVAILABLE:
    # Kafka source for real-time transactions
    transaction_stream_source = KafkaSource(
        name="transaction_stream_source",
        kafka_bootstrap_servers="kafka.insurance.internal:9092",
        topic="insurance.transactions",
        timestamp_field="event_timestamp",
        batch_source=transaction_source,  # Batch source for historical data
        message_format=JsonFormat(
            schema_json="""
            {
                "type": "record",
                "name": "Transaction",
                "fields": [
                    {"name": "transaction_id", "type": "string"},
                    {"name": "customer_id", "type": "string"},
                    {"name": "amount", "type": "double"},
                    {"name": "transaction_type", "type": "string"},
                    {"name": "merchant_category", "type": "string"},
                    {"name": "event_timestamp", "type": "string"}
                ]
            }
            """
        ),
        watermark_delay_threshold=timedelta(minutes=5),
        description="Kafka stream for real-time transaction data (DSS)",
        tags={"domain": "streaming", "use_case": "fraud_detection"},
    )

    # Kafka source for real-time claims events
    claims_stream_source = KafkaSource(
        name="claims_stream_source",
        kafka_bootstrap_servers="kafka.insurance.internal:9092",
        topic="insurance.claims.events",
        timestamp_field="event_timestamp",
        batch_source=claims_history_source,
        message_format=JsonFormat(
            schema_json="""
            {
                "type": "record",
                "name": "ClaimEvent",
                "fields": [
                    {"name": "claim_id", "type": "string"},
                    {"name": "customer_id", "type": "string"},
                    {"name": "claim_amount", "type": "double"},
                    {"name": "claim_type", "type": "string"},
                    {"name": "status", "type": "string"},
                    {"name": "event_timestamp", "type": "string"}
                ]
            }
            """
        ),
        watermark_delay_threshold=timedelta(minutes=5),
        description="Kafka stream for real-time claims events (DSS)",
        tags={"domain": "streaming", "use_case": "claims_processing"},
    )
