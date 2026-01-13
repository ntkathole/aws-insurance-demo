"""
Streaming Feature Views - Future DSS (Decision Support System)

These feature views are designed for future streaming use cases.
They support real-time transaction monitoring and continuous risk assessment
using StreamFeatureView with Kafka sources.

Transaction Type: Streaming (DSS - Future needs)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.aggregation import Aggregation
from feast.types import Float32, Float64, Int32, Int64, String, Bool

# Import entities and data sources
import sys
sys.path.insert(0, '..')
from entities import customer, transaction
from data_sources import (
    transaction_source,
    transaction_file_source,
    transaction_push_source,
    KAFKA_AVAILABLE,
)

# StreamFeatureView requires Kafka and specific Feast configuration
# Disabled for this demo - enable when Kafka infrastructure is ready
STREAM_FV_AVAILABLE = False


# =============================================================================
# TRANSACTION FEATURES (Batch-based)
# =============================================================================
# Transaction-level features for fraud detection - batch version
transaction_features_fv = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(hours=1),
    schema=[
        # Transaction identification
        Field(name="customer_id", dtype=String, description="Customer ID"),
        Field(name="merchant_id", dtype=String, description="Merchant ID"),
        
        # Transaction details
        Field(name="amount", dtype=Float64, description="Transaction amount"),
        Field(name="currency", dtype=String, description="Currency code"),
        Field(name="transaction_type", dtype=String, description="Transaction type"),
        Field(name="channel", dtype=String, description="Transaction channel"),
        
        # Merchant info
        Field(name="merchant_category", dtype=String, description="Merchant category code"),
        Field(name="merchant_country", dtype=String, description="Merchant country"),
        Field(name="merchant_risk_score", dtype=Float32, description="Merchant risk score"),
        
        # Device/Location
        Field(name="device_type", dtype=String, description="Device type"),
        Field(name="device_fingerprint_match", dtype=Bool, description="Device fingerprint matches"),
        Field(name="ip_country", dtype=String, description="IP country"),
        Field(name="distance_from_home", dtype=Float32, description="Distance from home location"),
        
        # Velocity indicators
        Field(name="is_first_transaction", dtype=Bool, description="First transaction for customer"),
        Field(name="minutes_since_last_txn", dtype=Int32, description="Minutes since last transaction"),
        Field(name="txn_count_last_hour", dtype=Int32, description="Transactions in last hour"),
        Field(name="txn_count_last_24h", dtype=Int32, description="Transactions in last 24 hours"),
        
        # Risk indicators
        Field(name="risk_score", dtype=Float32, description="Transaction risk score"),
        Field(name="fraud_probability", dtype=Float32, description="Fraud probability"),
        Field(name="high_risk_merchant_flag", dtype=Bool, description="High risk merchant"),
        Field(name="unusual_amount_flag", dtype=Bool, description="Unusual amount for customer"),
        Field(name="unusual_time_flag", dtype=Bool, description="Unusual time of day"),
        Field(name="unusual_location_flag", dtype=Bool, description="Unusual location"),
    ],
    online=True,
    source=transaction_source,
    tags={
        "domain": "streaming",
        "use_case": "fraud_detection",
        "latency_requirement": "real_time",
    },
    description="Transaction features for fraud detection",
)


# =============================================================================
# CUSTOMER TRANSACTION AGGREGATION FEATURES (DISABLED)
# =============================================================================
# NOTE: This feature view is disabled because it requires pre-computed 
# aggregation columns that need a streaming pipeline or batch job to populate.
# Enable when you have a separate customer_transaction_aggregations table.
#
# customer_transaction_agg_fv = FeatureView(
#     name="customer_transaction_aggregations",
#     entities=[customer],
#     ...
# )


# =============================================================================
# STREAM FEATURE VIEW - Transaction Stream (Future DSS)
# =============================================================================
# Streaming feature view with aggregations for real-time fraud detection
# NOTE: This requires Kafka to be configured and feast[kafka] installed

if STREAM_FV_AVAILABLE and KAFKA_AVAILABLE:
    from data_sources import transaction_stream_source
    
    @stream_feature_view(
        entities=[customer],
        ttl=timedelta(days=1),
        mode="pandas",
        schema=[
            Field(name="txn_count_1h_stream", dtype=Int64),
            Field(name="txn_amount_sum_1h_stream", dtype=Float64),
            Field(name="txn_amount_avg_1h_stream", dtype=Float64),
            Field(name="txn_count_24h_stream", dtype=Int64),
            Field(name="txn_amount_sum_24h_stream", dtype=Float64),
        ],
        timestamp_field="event_timestamp",
        aggregations=[
            # 1-hour aggregations
            Aggregation(
                column="amount",
                function="count",
                time_window=timedelta(hours=1),
            ),
            Aggregation(
                column="amount",
                function="sum",
                time_window=timedelta(hours=1),
            ),
            Aggregation(
                column="amount",
                function="avg",
                time_window=timedelta(hours=1),
            ),
            # 24-hour aggregations
            Aggregation(
                column="amount",
                function="count",
                time_window=timedelta(hours=24),
            ),
            Aggregation(
                column="amount",
                function="sum",
                time_window=timedelta(hours=24),
            ),
        ],
        online=True,
        source=transaction_stream_source,
        enable_tiling=True,  # Enable tiling for efficient aggregations
        tiling_hop_size=timedelta(minutes=5),  # 5-minute hop for tiling
        tags={
            "domain": "streaming",
            "use_case": "fraud_detection",
            "real_time": "true",
            "dss": "true",
        },
        description="Real-time transaction aggregations from Kafka stream (DSS)",
    )
    def transaction_stream_fv(df):
        """Transform incoming stream data if needed."""
        # Add any stream-specific transformations here
        return df
