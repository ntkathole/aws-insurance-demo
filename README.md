# AWS Insurance Demo - Feast Feature Store

![Feast Logo](https://raw.githubusercontent.com/feast-dev/feast/master/docs/assets/feast_logo.png)

A comprehensive Feast feature store example demonstrating **real-time auto underwriting** and **batch claims optimization** use cases for insurance/financial services, with AWS infrastructure (DynamoDB, Redshift, S3).

## 🎯 Use Cases

### 1. Auto Underwriting (Real-Time - PCM)
Real-time risk assessment and policy pricing using:
- Customer profile features
- Credit and financial indicators
- Risk scoring with on-demand transformations
- Sub-second latency requirements

### 2. Claims Optimization (Batch - Claims & Labs)
Batch processing for claims analysis:
- Historical claims patterns
- Lab results and medical indicators
- Fraud detection features
- Provider network analysis

### 3. Streaming Features (DSS - Future)
Prepared infrastructure for streaming:
- Real-time transaction monitoring
- Continuous risk assessment
- Event-driven feature updates

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AWS Infrastructure                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   S3 Bucket  │    │   Redshift   │    │   DynamoDB   │                  │
│  │  (Registry)  │    │(Offline Store)│   │(Online Store)│                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                               │
│                    ┌────────┴────────┐                                      │
│                    │  Feast Feature  │                                      │
│                    │     Server      │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│         ┌───────────────────┼───────────────────┐                           │
│         │                   │                   │                           │
│  ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐                     │
│  │  Real-Time  │    │    Batch    │    │  Streaming  │                     │
│  │Underwriting │    │   Claims    │    │    (DSS)    │                     │
│  │    (PCM)    │    │  Processing │    │   Future    │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
aws-insurance-demo/
├── README.md
├── requirements.txt
├── feature_repo/
│   ├── feature_store.yaml          # Feast configuration (DynamoDB/Redshift/S3)
│   ├── __init__.py
│   ├── entities.py                 # Entity definitions
│   ├── data_sources.py             # Redshift data sources
│   ├── feature_views/
│   │   ├── __init__.py
│   │   ├── underwriting_features.py   # Real-time underwriting (PCM)
│   │   ├── claims_features.py         # Batch claims optimization
│   │   └── streaming_features.py      # Future DSS streaming
│   ├── on_demand_features.py       # On-demand transformations
│   └── feature_services.py         # Feature services
├── scripts/
│   ├── setup_redshift_data.py      # Single script: generate & load data (Redshift or local)
│   ├── benchmark_online_server.py  # Latency testing script
│   └── setup_redshift_tables.sql   # SQL to create Redshift tables (reference)
├── notebooks/
│   ├── 01_setup_and_data_prep.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_latency_testing.ipynb
└── data/
    └── sample/                     # Sample parquet files for local testing
```

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with access to:
   - Amazon Redshift (cluster or serverless)
   - Amazon DynamoDB
   - Amazon S3

2. **AWS Credentials** configured:
   ```bash
   aws configure
   # Or set environment variables:
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-west-2
   ```

3. **Python 3.9+** with virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Setup

#### Step 1: Configure feature_store.yaml with your AWS settings

```bash
cd feature_repo
# Edit feature_store.yaml with your actual values:
```

```yaml
# Update these values in feature_store.yaml:
registry:
  path: s3://YOUR-BUCKET/insurance-demo/registry.pb

offline_store:
  type: redshift
  region: us-west-2
  cluster_id: YOUR-CLUSTER-ID        # e.g., feast-insurance-cluster
  database: insurance_features
  user: YOUR-REDSHIFT-USER           # e.g., feast_user
  s3_staging_location: s3://YOUR-BUCKET/insurance-demo/staging
  iam_role: arn:aws:iam::YOUR-ACCOUNT:role/YOUR-REDSHIFT-ROLE

online_store:
  type: dynamodb
  region: us-west-2
```

#### Step 2: Set up data

The `setup_redshift_data.py` script handles all data setup in one command:
- Creates the insurance schema and all tables (Redshift mode)
- Generates realistic sample data
- Loads data directly into Redshift OR generates local files for testing

```bash
cd scripts

# OPTION A: Local testing (no AWS required)
python setup_redshift_data.py --local-only --output-dir ../data/sample --num-customers 1000

# OPTION B: Redshift with environment variables
export REDSHIFT_HOST=YOUR-CLUSTER.xxxxx.us-west-2.redshift.amazonaws.com
export REDSHIFT_DATABASE=insurance_features
export REDSHIFT_USER=feast_user
export REDSHIFT_PASSWORD=YOUR-PASSWORD

python setup_redshift_data.py --num-customers 10000

# OPTION C: Redshift with command line arguments
python setup_redshift_data.py \
    --host YOUR-CLUSTER.xxxxx.us-west-2.redshift.amazonaws.com \
    --database insurance_features \
    --user feast_user \
    --password YOUR-PASSWORD \
    --num-customers 10000

# OPTION D: For large datasets (100K+ customers), use S3 COPY for faster loading
python setup_redshift_data.py \
    --num-customers 100000 \
    --use-s3 \
    --s3-bucket YOUR-BUCKET \
    --iam-role arn:aws:iam::YOUR-ACCOUNT:role/YOUR-REDSHIFT-ROLE
```

#### Step 3: Apply Feast definitions

```bash
cd ../feature_repo
feast apply
```

#### Step 4: Materialize features to DynamoDB

```bash
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

#### Step 5: Start the feature server

```bash
feast serve -h 0.0.0.0 -p 6566
```

## 📊 Features Overview

### Entity Definitions

| Entity | Description | Join Key |
|--------|-------------|----------|
| `customer` | Insurance customer/policyholder | `customer_id` |
| `policy` | Insurance policy | `policy_id` |
| `claim` | Insurance claim | `claim_id` |
| `provider` | Healthcare/service provider | `provider_id` |

### Feature Views

#### Real-Time Underwriting (PCM)
| Feature View | Description | TTL |
|--------------|-------------|-----|
| `customer_profile_features` | Demographics, history | 1 day |
| `customer_credit_features` | Credit scores, financial indicators | 1 hour |
| `customer_risk_features` | Risk metrics, behavioral patterns | 1 hour |
| `policy_features` | Policy details, coverage info | 1 day |

#### Batch Claims (Claims & Labs)
| Feature View | Description | TTL |
|--------------|-------------|-----|
| `claims_history_features` | Historical claims data | 7 days |
| `claims_aggregation_features` | Aggregated claims metrics | 1 day |
| `lab_results_features` | Medical lab indicators | 7 days |
| `provider_features` | Provider network data | 1 day |

#### On-Demand Features
| Feature View | Description |
|--------------|-------------|
| `underwriting_risk_score` | Real-time risk calculation |
| `claims_fraud_indicators` | Fraud detection signals |
| `policy_premium_adjustments` | Dynamic premium calculation |

### Aggregation Features

```python
# Example: Transaction aggregations with time windows
aggregations=[
    Aggregation(column="transaction_amount", function="sum", time_window=timedelta(hours=24)),
    Aggregation(column="transaction_amount", function="avg", time_window=timedelta(hours=24)),
    Aggregation(column="transaction_count", function="count", time_window=timedelta(hours=1)),
]
```

## 🧪 Latency Testing

### Quick Benchmark
```bash
python scripts/benchmark_online_server.py \
    --server-url http://localhost:6566 \
    --feature-service underwriting_v1 \
    --entity-key customer_id \
    --profile medium
```

### Comprehensive Benchmark
```bash
python scripts/benchmark_online_server.py \
    --server-url http://localhost:6566 \
    --features "customer_profile_features:age,customer_profile_features:income,customer_risk_features:risk_score" \
    --entity-key customer_id \
    --batch-sizes 1,10,50,100,500 \
    --num-requests 1000 \
    --concurrency 20 \
    --output results.json
```

### Expected Latency Targets

| Batch Size | Target P99 (ms) | Use Case |
|------------|-----------------|----------|
| 1 | < 10 | Single policy quote |
| 10 | < 25 | Small batch processing |
| 100 | < 100 | Bulk underwriting |
| 500 | < 300 | Large batch jobs |

## 🔧 Configuration Options

### DynamoDB Online Store Options
```yaml
online_store:
  type: dynamodb
  region: us-west-2
  consistent_reads: false          # Set to true for strong consistency
  batch_size: 40                   # Items per BatchGetItem call
  table_name_template: "{project}.{table_name}"
```

### Redshift Offline Store Options
```yaml
offline_store:
  type: redshift
  region: us-west-2
  cluster_id: feast-cluster        # For provisioned clusters
  # workgroup: feast-workgroup     # For serverless
  database: feast_db
  user: feast_user
  s3_staging_location: s3://your-bucket/feast/staging
  iam_role: arn:aws:iam::123456789012:role/RedshiftS3AccessRole
```

### S3 Registry Options
```yaml
registry:
  path: s3://your-bucket/feast/registry.pb
  cache_ttl_seconds: 60
```

## 📈 Scaling Considerations

### For High-Volume Real-Time (PCM)
1. **DynamoDB**: Enable auto-scaling, consider DAX caching
2. **Feature Server**: Deploy multiple instances behind ALB
3. **Batch Size**: Optimize for typical request patterns

### For Large Batch Processing (Claims & Labs)
1. **Redshift**: Use larger cluster or serverless with auto-scaling
2. **S3 Staging**: Use lifecycle policies for staging data
3. **Materialization**: Schedule during off-peak hours

### For Future Streaming (DSS)
1. **Kafka/Kinesis**: Configure appropriate partitions
2. **Stream Feature Views**: Enable tiling for efficient aggregations
3. **Push Source**: For real-time feature updates

## 🔒 Security Best Practices

1. **IAM Roles**: Use least-privilege policies
2. **VPC**: Deploy in private subnets
3. **Encryption**: Enable at-rest and in-transit encryption
4. **Secrets**: Use AWS Secrets Manager for credentials

## 📚 Additional Resources

- [Feast Documentation](https://docs.feast.dev/)
- [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)
- [Redshift Performance Tuning](https://docs.aws.amazon.com/redshift/latest/dg/c_best-practices-best-dist-key.html)
- [Feast Benchmarks](https://docs.feast.dev/blog/feast-benchmarks)

## 🤝 Contributing

Contributions are welcome! Please see the main [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details.
