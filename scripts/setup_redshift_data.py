#!/usr/bin/env python
"""
AWS Insurance Demo - Data Setup Script

This single script handles all data setup needs for the demo:
1. Creates the insurance schema and all required tables in Redshift
2. Generates realistic sample data
3. Loads data directly into Redshift (no local file generation needed)

Modes:
- Redshift mode (default): Generates and loads data directly to Redshift
- Local mode (--local-only): Generates parquet files locally for testing without AWS

Usage:
    # MODE 1: Load directly to Redshift (recommended for demo)
    export REDSHIFT_HOST=my-cluster.xxxxx.us-west-2.redshift.amazonaws.com
    export REDSHIFT_DATABASE=insurance_features
    export REDSHIFT_USER=feast_user
    export REDSHIFT_PASSWORD=your_password
    python setup_redshift_data.py --num-customers 10000

    # MODE 2: Local testing only (no Redshift required)
    python setup_redshift_data.py --local-only --output-dir ../data/sample --num-customers 1000

    # MODE 3: For faster Redshift loading with large datasets, use S3:
    python setup_redshift_data.py \
        --host my-cluster.xxxxx.us-west-2.redshift.amazonaws.com \
        --num-customers 100000 \
        --use-s3 \
        --s3-bucket my-feast-bucket \
        --iam-role arn:aws:iam::123456789012:role/RedshiftS3Role

Requirements:
    pip install redshift-connector pandas numpy boto3 pyarrow
"""

import argparse
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import redshift_connector
try:
    import redshift_connector
except ImportError:
    print("Error: redshift_connector is required.")
    print("Install with: pip install redshift-connector")
    sys.exit(1)

# Try to import boto3 for S3 operations
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# =============================================================================
# TABLE DEFINITIONS (DDL)
# =============================================================================

CREATE_SCHEMA_SQL = "CREATE SCHEMA IF NOT EXISTS insurance;"

TABLE_DEFINITIONS = {
    "insurance.customer_profiles": """
        DROP TABLE IF EXISTS insurance.customer_profiles;
        CREATE TABLE insurance.customer_profiles (
            customer_id VARCHAR(20) NOT NULL,
            age INT,
            gender VARCHAR(10),
            marital_status VARCHAR(20),
            occupation VARCHAR(50),
            education_level VARCHAR(50),
            state VARCHAR(2),
            zip_code VARCHAR(10),
            urban_rural VARCHAR(20),
            region_risk_zone INT,
            customer_tenure_months INT,
            num_policies INT,
            loyalty_tier VARCHAR(20),
            has_agent BOOLEAN,
            event_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY (customer_id)
        )
        DISTKEY(customer_id)
        SORTKEY(event_timestamp);
    """,
    
    "insurance.customer_credit_data": """
        DROP TABLE IF EXISTS insurance.customer_credit_data;
        CREATE TABLE insurance.customer_credit_data (
            customer_id VARCHAR(20) NOT NULL,
            credit_score INT,
            credit_score_tier VARCHAR(5),
            credit_score_change_3m INT,
            credit_history_length_months INT,
            num_credit_accounts INT,
            num_delinquencies INT,
            bankruptcy_flag BOOLEAN,
            annual_income DECIMAL(15, 2),
            debt_to_income_ratio DECIMAL(5, 2),
            payment_history_score DECIMAL(5, 1),
            insurance_score INT,
            prior_coverage_lapse BOOLEAN,
            event_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY (customer_id)
        )
        DISTKEY(customer_id)
        SORTKEY(event_timestamp);
    """,
    
    "insurance.customer_risk_metrics": """
        DROP TABLE IF EXISTS insurance.customer_risk_metrics;
        CREATE TABLE insurance.customer_risk_metrics (
            customer_id VARCHAR(20) NOT NULL,
            overall_risk_score DECIMAL(5, 1),
            claims_risk_score DECIMAL(5, 1),
            fraud_risk_score DECIMAL(5, 1),
            churn_risk_score DECIMAL(5, 1),
            num_claims_1y INT,
            num_claims_3y INT,
            total_claims_amount_1y DECIMAL(15, 2),
            avg_claim_amount DECIMAL(15, 2),
            policy_changes_1y INT,
            late_payments_1y INT,
            inquiry_count_30d INT,
            driving_violations_3y INT,
            at_fault_accidents_3y INT,
            dui_flag BOOLEAN,
            risk_segment VARCHAR(20),
            underwriting_tier VARCHAR(20),
            event_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY (customer_id)
        )
        DISTKEY(customer_id)
        SORTKEY(event_timestamp);
    """,
    
    "insurance.claims_history": """
        DROP TABLE IF EXISTS insurance.claims_history;
        CREATE TABLE insurance.claims_history (
            claim_id VARCHAR(20) NOT NULL,
            customer_id VARCHAR(20) NOT NULL,
            policy_id VARCHAR(20),
            provider_id VARCHAR(20),
            claim_type VARCHAR(30),
            claim_subtype VARCHAR(30),
            claim_status VARCHAR(20),
            claim_amount_requested DECIMAL(15, 2),
            claim_amount_approved DECIMAL(15, 2),
            claim_amount_paid DECIMAL(15, 2),
            days_to_first_contact INT,
            days_to_settlement INT,
            claim_age_days INT,
            injury_flag BOOLEAN,
            fatality_flag BOOLEAN,
            litigation_flag BOOLEAN,
            attorney_represented BOOLEAN,
            num_parties_involved INT,
            siu_referral BOOLEAN,
            fraud_score DECIMAL(5, 1),
            suspicious_indicators_count INT,
            adjuster_id VARCHAR(20),
            adjuster_workload INT,
            priority_score DECIMAL(5, 1),
            event_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY (claim_id)
        )
        DISTKEY(customer_id)
        SORTKEY(event_timestamp);
    """,
    
    "insurance.transactions": """
        DROP TABLE IF EXISTS insurance.transactions;
        CREATE TABLE insurance.transactions (
            transaction_id VARCHAR(20) NOT NULL,
            customer_id VARCHAR(20) NOT NULL,
            merchant_id VARCHAR(20),
            amount DECIMAL(15, 2),
            currency VARCHAR(5),
            transaction_type VARCHAR(20),
            channel VARCHAR(20),
            merchant_category VARCHAR(50),
            merchant_country VARCHAR(50),
            merchant_risk_score DECIMAL(5, 2),
            device_type VARCHAR(20),
            device_fingerprint_match BOOLEAN,
            ip_country VARCHAR(50),
            distance_from_home DECIMAL(10, 1),
            is_first_transaction BOOLEAN,
            minutes_since_last_txn INT,
            txn_count_last_hour INT,
            txn_count_last_24h INT,
            risk_score DECIMAL(5, 1),
            fraud_probability DECIMAL(5, 3),
            high_risk_merchant_flag BOOLEAN,
            unusual_amount_flag BOOLEAN,
            unusual_time_flag BOOLEAN,
            unusual_location_flag BOOLEAN,
            event_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP,
            PRIMARY KEY (transaction_id)
        )
        DISTKEY(customer_id)
        SORTKEY(event_timestamp);
    """,
}


# =============================================================================
# DATA GENERATOR
# =============================================================================

class InsuranceDataGenerator:
    """Generate realistic insurance domain sample data."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
        
        self.occupations = [
            'Engineer', 'Teacher', 'Nurse', 'Manager', 'Sales', 'Technician',
            'Driver', 'Construction', 'Retail', 'Finance', 'Healthcare', 'IT',
            'Manufacturing', 'Government', 'Retired', 'Student', 'Self-employed'
        ]
        
        self.education_levels = [
            'High School', 'Some College', 'Associate', 'Bachelor', 'Master', 'Doctorate'
        ]
        
        self.claim_types = [
            'Collision', 'Comprehensive', 'Liability', 'Medical', 'Property',
            'Theft', 'Weather', 'Glass', 'Towing', 'Rental'
        ]
    
    def _generate_timestamps(self, n: int, start_date: datetime, end_date: datetime) -> List[datetime]:
        delta = end_date - start_date
        return [
            start_date + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
            for _ in range(n)
        ]
    
    def generate_customer_profiles(self, num_customers: int, end_date: datetime) -> pd.DataFrame:
        print(f"  Generating {num_customers} customer profiles...")
        
        customer_ids = [f"CUST{i:08d}" for i in range(1, num_customers + 1)]
        ages = np.clip(np.random.normal(42, 15, num_customers), 18, 85).astype(int)
        max_tenure = (ages - 18) * 6
        tenure_months = [random.randint(1, max(1, mt)) for mt in max_tenure]
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': [random.choice(['M', 'F', 'O']) for _ in range(num_customers)],
            'marital_status': [random.choice(['Single', 'Married', 'Divorced', 'Widowed']) for _ in range(num_customers)],
            'occupation': [random.choice(self.occupations) for _ in range(num_customers)],
            'education_level': [random.choice(self.education_levels) for _ in range(num_customers)],
            'state': [random.choice(self.states) for _ in range(num_customers)],
            'zip_code': [f"{random.randint(10000, 99999)}" for _ in range(num_customers)],
            'urban_rural': [random.choices(['Urban', 'Suburban', 'Rural'], weights=[0.4, 0.4, 0.2])[0] for _ in range(num_customers)],
            'region_risk_zone': [random.randint(1, 5) for _ in range(num_customers)],
            'customer_tenure_months': tenure_months,
            'num_policies': [random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0] for _ in range(num_customers)],
            'loyalty_tier': [random.choices(['Bronze', 'Silver', 'Gold', 'Platinum'], weights=[0.4, 0.35, 0.2, 0.05])[0] for _ in range(num_customers)],
            'has_agent': [random.random() > 0.3 for _ in range(num_customers)],
            'event_timestamp': self._generate_timestamps(num_customers, end_date - timedelta(days=30), end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 365)) for _ in range(num_customers)],
        })
    
    def generate_customer_credit(self, customer_ids: List[str], end_date: datetime) -> pd.DataFrame:
        print(f"  Generating credit data for {len(customer_ids)} customers...")
        num = len(customer_ids)
        
        credit_scores = np.clip(np.random.normal(700, 80, num), 300, 850).astype(int)
        
        def score_to_tier(s):
            if s >= 800: return 'A'
            elif s >= 740: return 'B'
            elif s >= 670: return 'C'
            elif s >= 580: return 'D'
            return 'F'
        
        insurance_scores = np.clip(credit_scores + np.random.normal(0, 30, num), 300, 900).astype(int)
        incomes = np.clip(credit_scores * 100 + np.random.normal(20000, 15000, num), 15000, 500000).astype(int)
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'credit_score': credit_scores,
            'credit_score_tier': [score_to_tier(s) for s in credit_scores],
            'credit_score_change_3m': [random.randint(-30, 30) for _ in range(num)],
            'credit_history_length_months': [random.randint(12, 360) for _ in range(num)],
            'num_credit_accounts': [random.randint(1, 20) for _ in range(num)],
            'num_delinquencies': [random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0] for _ in range(num)],
            'bankruptcy_flag': [random.random() < 0.02 for _ in range(num)],
            'annual_income': incomes.astype(float),
            'debt_to_income_ratio': [round(random.uniform(0.1, 0.6), 2) for _ in range(num)],
            'payment_history_score': [round(random.uniform(60, 100), 1) for _ in range(num)],
            'insurance_score': insurance_scores,
            'prior_coverage_lapse': [random.random() < 0.1 for _ in range(num)],
            'event_timestamp': self._generate_timestamps(num, end_date - timedelta(days=7), end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 30)) for _ in range(num)],
        })
    
    def generate_customer_risk(self, customer_ids: List[str], credit_df: pd.DataFrame, end_date: datetime) -> pd.DataFrame:
        print(f"  Generating risk data for {len(customer_ids)} customers...")
        num = len(customer_ids)
        
        credit_lookup = dict(zip(credit_df['customer_id'], credit_df['credit_score']))
        
        risk_scores = []
        for cid in customer_ids:
            credit_score = credit_lookup.get(cid, 700)
            base_risk = 100 - (credit_score - 300) / 5.5
            risk = np.clip(base_risk + np.random.normal(0, 15), 0, 100)
            risk_scores.append(round(risk, 1))
        
        claims_1y = np.random.poisson(0.3, num)
        claims_3y = claims_1y + np.random.poisson(0.5, num)
        
        def risk_to_segment(s):
            if s < 40: return 'low'
            elif s < 70: return 'medium'
            return 'high'
        
        def risk_to_tier(s):
            if s < 30: return 'preferred'
            elif s < 50: return 'standard'
            elif s < 70: return 'substandard'
            return 'decline'
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'overall_risk_score': risk_scores,
            'claims_risk_score': [round(r * random.uniform(0.8, 1.2), 1) for r in risk_scores],
            'fraud_risk_score': [round(random.uniform(0, 30), 1) for _ in range(num)],
            'churn_risk_score': [round(random.uniform(5, 60), 1) for _ in range(num)],
            'num_claims_1y': claims_1y.tolist(),
            'num_claims_3y': claims_3y.tolist(),
            'total_claims_amount_1y': [c * random.uniform(1000, 15000) for c in claims_1y],
            'avg_claim_amount': [random.uniform(500, 10000) for _ in range(num)],
            'policy_changes_1y': [random.randint(0, 3) for _ in range(num)],
            'late_payments_1y': [random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0] for _ in range(num)],
            'inquiry_count_30d': [random.randint(0, 5) for _ in range(num)],
            'driving_violations_3y': [random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0] for _ in range(num)],
            'at_fault_accidents_3y': [random.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0] for _ in range(num)],
            'dui_flag': [random.random() < 0.01 for _ in range(num)],
            'risk_segment': [risk_to_segment(r) for r in risk_scores],
            'underwriting_tier': [risk_to_tier(r) for r in risk_scores],
            'event_timestamp': self._generate_timestamps(num, end_date - timedelta(days=1), end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 7)) for _ in range(num)],
        })
    
    def generate_claims_history(self, customer_ids: List[str], num_claims: int, end_date: datetime) -> pd.DataFrame:
        print(f"  Generating {num_claims} claims...")
        
        claim_ids = [f"CLM{i:010d}" for i in range(1, num_claims + 1)]
        
        customer_weights = np.random.exponential(1, len(customer_ids))
        customer_weights /= customer_weights.sum()
        claim_customers = np.random.choice(customer_ids, num_claims, p=customer_weights)
        
        claim_amounts_requested = np.clip(np.random.lognormal(8, 1.2, num_claims), 100, 500000)
        approval_ratios = np.clip(np.random.beta(8, 2, num_claims), 0, 1)
        claim_amounts_approved = claim_amounts_requested * approval_ratios
        
        statuses = ['Approved', 'Denied', 'Pending', 'Under Review', 'Closed']
        status_weights = [0.5, 0.1, 0.1, 0.15, 0.15]
        
        return pd.DataFrame({
            'claim_id': claim_ids,
            'customer_id': claim_customers,
            'policy_id': [f"POL{i:08d}" for i in np.random.randint(1, len(customer_ids), num_claims)],
            'provider_id': [f"PROV{i:06d}" for i in np.random.randint(1, 1000, num_claims)],
            'claim_type': [random.choice(self.claim_types) for _ in range(num_claims)],
            'claim_subtype': [f"Subtype_{random.randint(1, 5)}" for _ in range(num_claims)],
            'claim_status': [random.choices(statuses, weights=status_weights)[0] for _ in range(num_claims)],
            'claim_amount_requested': claim_amounts_requested.round(2).tolist(),
            'claim_amount_approved': claim_amounts_approved.round(2).tolist(),
            'claim_amount_paid': (claim_amounts_approved * np.random.uniform(0.9, 1.0, num_claims)).round(2).tolist(),
            'days_to_first_contact': [random.randint(0, 5) for _ in range(num_claims)],
            'days_to_settlement': [random.randint(5, 90) for _ in range(num_claims)],
            'claim_age_days': [random.randint(0, 365) for _ in range(num_claims)],
            'injury_flag': [random.random() < 0.15 for _ in range(num_claims)],
            'fatality_flag': [random.random() < 0.001 for _ in range(num_claims)],
            'litigation_flag': [random.random() < 0.05 for _ in range(num_claims)],
            'attorney_represented': [random.random() < 0.08 for _ in range(num_claims)],
            'num_parties_involved': [random.randint(1, 4) for _ in range(num_claims)],
            'siu_referral': [random.random() < 0.03 for _ in range(num_claims)],
            'fraud_score': [round(random.uniform(0, 50), 1) for _ in range(num_claims)],
            'suspicious_indicators_count': [random.choices([0, 1, 2, 3], weights=[0.8, 0.12, 0.05, 0.03])[0] for _ in range(num_claims)],
            'adjuster_id': [f"ADJ{i:04d}" for i in np.random.randint(1, 100, num_claims)],
            'adjuster_workload': [random.randint(10, 50) for _ in range(num_claims)],
            'priority_score': [round(random.uniform(1, 10), 1) for _ in range(num_claims)],
            'event_timestamp': self._generate_timestamps(num_claims, end_date - timedelta(days=365), end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 365)) for _ in range(num_claims)],
        })
    
    def generate_transactions(self, customer_ids: List[str], num_transactions: int, end_date: datetime) -> pd.DataFrame:
        print(f"  Generating {num_transactions} transactions...")
        
        transaction_ids = [f"TXN{i:012d}" for i in range(1, num_transactions + 1)]
        
        customer_weights = np.random.exponential(1, len(customer_ids))
        customer_weights /= customer_weights.sum()
        txn_customers = np.random.choice(customer_ids, num_transactions, p=customer_weights)
        
        amounts = np.clip(np.random.lognormal(4, 1.5, num_transactions), 1, 50000)
        
        merchant_categories = ['Grocery', 'Gas', 'Restaurant', 'Retail', 'Online', 'Travel', 'Entertainment', 'Healthcare', 'Utilities', 'Insurance']
        channels = ['Online', 'POS', 'ATM', 'Mobile', 'Phone']
        
        return pd.DataFrame({
            'transaction_id': transaction_ids,
            'customer_id': txn_customers,
            'merchant_id': [f"MERCH{i:06d}" for i in np.random.randint(1, 10000, num_transactions)],
            'amount': amounts.round(2).tolist(),
            'currency': ['USD'] * num_transactions,
            'transaction_type': [random.choice(['Purchase', 'Refund', 'Transfer', 'Payment']) for _ in range(num_transactions)],
            'channel': [random.choices(channels, weights=[0.35, 0.3, 0.1, 0.2, 0.05])[0] for _ in range(num_transactions)],
            'merchant_category': [random.choice(merchant_categories) for _ in range(num_transactions)],
            'merchant_country': [random.choices(['US', 'CA', 'UK', 'Other'], weights=[0.9, 0.04, 0.03, 0.03])[0] for _ in range(num_transactions)],
            'merchant_risk_score': [round(random.uniform(0, 0.5), 2) for _ in range(num_transactions)],
            'device_type': [random.choice(['Mobile', 'Desktop', 'Tablet', 'Unknown']) for _ in range(num_transactions)],
            'device_fingerprint_match': [random.random() > 0.05 for _ in range(num_transactions)],
            'ip_country': [random.choices(['US', 'CA', 'UK', 'Other'], weights=[0.9, 0.04, 0.03, 0.03])[0] for _ in range(num_transactions)],
            'distance_from_home': [round(random.uniform(0, 100), 1) for _ in range(num_transactions)],
            'is_first_transaction': [random.random() < 0.02 for _ in range(num_transactions)],
            'minutes_since_last_txn': [random.randint(1, 10000) for _ in range(num_transactions)],
            'txn_count_last_hour': [random.randint(0, 5) for _ in range(num_transactions)],
            'txn_count_last_24h': [random.randint(0, 20) for _ in range(num_transactions)],
            'risk_score': [round(random.uniform(0, 100), 1) for _ in range(num_transactions)],
            'fraud_probability': [round(random.uniform(0, 0.3), 3) for _ in range(num_transactions)],
            'high_risk_merchant_flag': [random.random() < 0.05 for _ in range(num_transactions)],
            'unusual_amount_flag': [random.random() < 0.1 for _ in range(num_transactions)],
            'unusual_time_flag': [random.random() < 0.08 for _ in range(num_transactions)],
            'unusual_location_flag': [random.random() < 0.05 for _ in range(num_transactions)],
            'event_timestamp': self._generate_timestamps(num_transactions, end_date - timedelta(days=30), end_date),
            'created_at': [end_date - timedelta(days=random.randint(0, 30)) for _ in range(num_transactions)],
        })


# =============================================================================
# REDSHIFT OPERATIONS
# =============================================================================

def get_connection(
    host: str = None,
    database: str = None,
    user: str = None,
    password: str = None,
    port: int = 5439,
    iam: bool = False,
    cluster_identifier: str = None,
    region: str = "us-east-1",
    profile: str = None,
    role_arn: str = None,
    access_key_id: str = None,
    secret_access_key: str = None,
    session_token: str = None,
):
    """Create a Redshift connection (supports password, IAM, and role-based auth)."""
    if iam:
        # IAM authentication - no password needed
        connect_params = {
            "iam": True,
            "database": database,
            "db_user": user,
            "cluster_identifier": cluster_identifier,
            "region": region,
        }
        
        # If role ARN provided, assume the role first
        if role_arn:
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required for IAM role assumption. Install with: pip install boto3")
            
            print(f"  Assuming IAM role: {role_arn}")
            sts_client = boto3.client('sts', region_name=region)
            assumed_role = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="FeastRedshiftSetup"
            )
            credentials = assumed_role['Credentials']
            connect_params["access_key_id"] = credentials['AccessKeyId']
            connect_params["secret_access_key"] = credentials['SecretAccessKey']
            connect_params["session_token"] = credentials['SessionToken']
        elif access_key_id and secret_access_key:
            # Use provided credentials directly
            connect_params["access_key_id"] = access_key_id
            connect_params["secret_access_key"] = secret_access_key
            if session_token:
                connect_params["session_token"] = session_token
        elif profile:
            connect_params["profile"] = profile
            
        print(f"  Using IAM authentication (cluster: {cluster_identifier}, region: {region})")
        return redshift_connector.connect(**connect_params)
    else:
        # Traditional password authentication
        return redshift_connector.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
        )


def create_tables(conn) -> None:
    """Create schema and all required tables."""
    print("\n" + "=" * 60)
    print("STEP 1: Creating schema and tables")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    # Create schema
    print("  Creating schema 'insurance'...")
    cursor.execute(CREATE_SCHEMA_SQL)
    conn.commit()
    
    # Create tables - execute each statement separately
    for table_name, ddl in TABLE_DEFINITIONS.items():
        print(f"  Creating table {table_name}...")
        # Split DDL into individual statements and execute each
        statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
        for stmt in statements:
            cursor.execute(stmt)
            conn.commit()
    
    cursor.close()
    print("  ✓ All tables created successfully!")


def prepare_dataframe_for_insert(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for Redshift insertion."""
    df = df.copy()
    
    # Convert timestamps to strings
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle NaN values
    df = df.fillna('')
    
    return df


def load_dataframe_direct(conn, table_name: str, df: pd.DataFrame, batch_size: int = 500) -> None:
    """Load DataFrame directly into Redshift using INSERT statements."""
    df = prepare_dataframe_for_insert(df)
    
    columns = list(df.columns)
    column_str = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"
    
    cursor = conn.cursor()
    total_rows = len(df)
    inserted = 0
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]
        values = [tuple(row) for row in batch.values]
        cursor.executemany(insert_sql, values)
        inserted += len(batch)
        print(f"    Inserted {inserted}/{total_rows} rows", end='\r')
    
    conn.commit()
    cursor.close()
    print(f"    Inserted {total_rows} rows                    ")


def load_dataframe_via_s3(
    conn,
    table_name: str,
    df: pd.DataFrame,
    s3_bucket: str,
    s3_prefix: str,
    iam_role: str,
    region: str = 'us-west-2',
) -> None:
    """Load DataFrame via S3 COPY (faster for large datasets)."""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required for S3 loading. Install with: pip install boto3")
    
    # Write DataFrame to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)
    
    try:
        # Upload to S3
        s3_client = boto3.client('s3', region_name=region)
        s3_key = f"{s3_prefix}/{table_name.replace('.', '_')}.parquet"
        
        print(f"    Uploading to s3://{s3_bucket}/{s3_key}...")
        s3_client.upload_file(tmp_path, s3_bucket, s3_key)
        
        # COPY from S3
        copy_sql = f"""
        COPY {table_name}
        FROM 's3://{s3_bucket}/{s3_key}'
        IAM_ROLE '{iam_role}'
        FORMAT AS PARQUET;
        """
        
        cursor = conn.cursor()
        cursor.execute(copy_sql)
        conn.commit()
        cursor.close()
        
        print(f"    Loaded {len(df)} rows via S3 COPY")
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def verify_data(conn) -> None:
    """Verify data was loaded correctly."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying data")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    for table_name in TABLE_DEFINITIONS.keys():
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"  {table_name}: {row_count:,} rows")
        except Exception as e:
            print(f"  {table_name}: Error - {e}")
    
    cursor.close()


def generate_local_files(num_customers: int, output_dir: str, seed: int = 42) -> dict:
    """Generate sample data and save to local parquet files for testing."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("LOCAL MODE: Generating sample data files")
    print("=" * 60)
    
    generator = InsuranceDataGenerator(seed=seed)
    end_date = datetime.now()
    
    # Generate customer profiles
    profiles_df = generator.generate_customer_profiles(num_customers, end_date)
    customer_ids = profiles_df['customer_id'].tolist()
    
    # Generate related data
    credit_df = generator.generate_customer_credit(customer_ids, end_date)
    risk_df = generator.generate_customer_risk(customer_ids, credit_df, end_date)
    
    num_claims = int(num_customers * 0.5)
    claims_df = generator.generate_claims_history(customer_ids, num_claims, end_date)
    
    num_transactions = num_customers * 50
    transactions_df = generator.generate_transactions(customer_ids, num_transactions, end_date)
    
    # Save to parquet files
    datasets = {
        'profiles': ('customer_profiles.parquet', profiles_df),
        'credit': ('customer_credit.parquet', credit_df),
        'risk': ('customer_risk.parquet', risk_df),
        'claims': ('claims_history.parquet', claims_df),
        'transactions': ('transactions.parquet', transactions_df),
    }
    
    print("\n  Saving files...")
    for name, (filename, df) in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df.to_parquet(filepath, index=False)
        print(f"    {filename}: {len(df):,} rows")
    
    print("\n" + "=" * 60)
    print("✓ LOCAL DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Customer profiles: {len(profiles_df):,} records")
    print(f"Customer credit: {len(credit_df):,} records")
    print(f"Customer risk: {len(risk_df):,} records")
    print(f"Claims history: {len(claims_df):,} records")
    print(f"Transactions: {len(transactions_df):,} records")
    print("\nNext steps for local testing:")
    print("  1. Update feature_store.yaml to use file sources")
    print("  2. Run: cd ../feature_repo && feast apply")
    print("  3. Run: feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)")
    print("=" * 60)
    
    return {
        'profiles': profiles_df,
        'credit': credit_df,
        'risk': risk_df,
        'claims': claims_df,
        'transactions': transactions_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Set up data for AWS Insurance Demo (Redshift or local files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MODE 1: Local testing only (no AWS required)
  python setup_redshift_data.py --local-only --output-dir ../data/sample --num-customers 1000

  # MODE 2: Load to Redshift with IAM authentication
  python setup_redshift_data.py \\
      --iam \\
      --cluster-identifier my-redshift-cluster \\
      --database feast_db \\
      --user feast_user \\
      --region us-east-1 \\
      --num-customers 10000

  # MODE 3: Load to Redshift with IAM role assumption
  python setup_redshift_data.py \\
      --iam \\
      --cluster-identifier my-redshift-cluster \\
      --database feast_db \\
      --user feast_user \\
      --region us-east-1 \\
      --role-arn arn:aws:iam::123456789012:role/RedshiftAccessRole \\
      --num-customers 10000

  # MODE 4: Load to Redshift with password authentication
  python setup_redshift_data.py \\
      --host my-cluster.xxxxx.us-west-2.redshift.amazonaws.com \\
      --database feast_db \\
      --user feast_user \\
      --password mypassword \\
      --num-customers 10000

  # MODE 5: For large datasets, use S3 COPY (faster)
  python setup_redshift_data.py \\
      --iam \\
      --cluster-identifier my-redshift-cluster \\
      --num-customers 100000 \\
      --use-s3 \\
      --s3-bucket my-feast-bucket \\
      --s3-iam-role arn:aws:iam::123456789012:role/RedshiftS3Role
        """
    )
    
    # Local mode arguments
    parser.add_argument("--local-only", action="store_true",
                       help="Generate local parquet files only (no Redshift required)")
    parser.add_argument("--output-dir", default="../data/sample",
                       help="Output directory for local parquet files (with --local-only)")
    
    # IAM authentication arguments
    parser.add_argument("--iam", action="store_true",
                       help="Use IAM authentication (no password required)")
    parser.add_argument("--cluster-identifier", default=os.environ.get("REDSHIFT_CLUSTER_ID"),
                       help="Redshift cluster identifier (for IAM auth)")
    parser.add_argument("--role-arn", default=os.environ.get("AWS_ROLE_ARN"),
                       help="IAM role ARN to assume for authentication")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"),
                       help="AWS profile name (optional, for IAM auth)")
    
    # Connection arguments (password auth)
    parser.add_argument("--host", default=os.environ.get("REDSHIFT_HOST"),
                       help="Redshift cluster endpoint (for password auth)")
    parser.add_argument("--database", default=os.environ.get("REDSHIFT_DATABASE", "feast_db"),
                       help="Redshift database name")
    parser.add_argument("--user", default=os.environ.get("REDSHIFT_USER"),
                       help="Redshift username")
    parser.add_argument("--password", default=os.environ.get("REDSHIFT_PASSWORD"),
                       help="Redshift password (for password auth)")
    parser.add_argument("--port", type=int, default=5439, help="Redshift port (default: 5439)")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), 
                       help="AWS region")
    
    # Data generation arguments
    parser.add_argument("--num-customers", type=int, default=10000,
                       help="Number of customers to generate (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # S3 arguments for faster loading
    parser.add_argument("--use-s3", action="store_true",
                       help="Use S3 COPY for faster loading (requires --s3-bucket and --s3-iam-role)")
    parser.add_argument("--s3-bucket", help="S3 bucket for staging data")
    parser.add_argument("--s3-prefix", default="feast-staging", help="S3 prefix for staging files")
    parser.add_argument("--s3-iam-role", help="IAM role ARN for Redshift to access S3")
    
    # Other options
    parser.add_argument("--skip-tables", action="store_true", help="Skip table creation (tables already exist)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    
    args = parser.parse_args()
    
    # Handle local-only mode
    if args.local_only:
        generate_local_files(args.num_customers, args.output_dir, args.seed)
        return
    
    # Validate required arguments for Redshift mode
    if args.iam:
        # IAM authentication
        if not args.cluster_identifier:
            parser.error("--cluster-identifier is required when using --iam")
        if not args.user:
            parser.error("--user is required (db_user for IAM auth)")
        if not args.database:
            parser.error("--database is required")
    else:
        # Password authentication
        if not args.host:
            parser.error("--host is required (or use --iam for IAM auth). Use --local-only for local testing.")
        if not args.user:
            parser.error("--user is required. Use --local-only for local testing.")
        if not args.password:
            parser.error("--password is required (or use --iam for IAM auth). Use --local-only for local testing.")
    
    if args.use_s3:
        if not args.s3_bucket:
            parser.error("--s3-bucket is required when using --use-s3")
        if not args.s3_iam_role:
            parser.error("--s3-iam-role is required when using --use-s3")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("AWS INSURANCE DEMO - REDSHIFT DATA SETUP")
    print("=" * 60)
    if args.iam:
        print(f"Auth: IAM (cluster: {args.cluster_identifier})")
        print(f"Region: {args.region}")
        if args.role_arn:
            print(f"Role ARN: {args.role_arn}")
        elif args.profile:
            print(f"Profile: {args.profile}")
    else:
        print(f"Host: {args.host}")
    print(f"Database: {args.database}")
    print(f"User: {args.user}")
    print(f"Customers to generate: {args.num_customers:,}")
    print(f"Loading method: {'S3 COPY' if args.use_s3 else 'Direct INSERT'}")
    
    # Connect to Redshift
    print("\nConnecting to Redshift...")
    conn = get_connection(
        host=args.host,
        database=args.database,
        user=args.user,
        password=args.password,
        port=args.port,
        iam=args.iam,
        cluster_identifier=args.cluster_identifier,
        region=args.region,
        profile=args.profile,
        role_arn=args.role_arn,
    )
    print("  ✓ Connected!")
    
    try:
        # Verify only mode
        if args.verify_only:
            verify_data(conn)
            return
        
        # Step 1: Create tables
        if not args.skip_tables:
            create_tables(conn)
        
        # Step 2: Generate and load data
        print("\n" + "=" * 60)
        print("STEP 2: Generating and loading data")
        print("=" * 60)
        
        generator = InsuranceDataGenerator(seed=args.seed)
        end_date = datetime.now()
        
        # Generate customer profiles
        profiles_df = generator.generate_customer_profiles(args.num_customers, end_date)
        customer_ids = profiles_df['customer_id'].tolist()
        
        # Generate related data
        credit_df = generator.generate_customer_credit(customer_ids, end_date)
        risk_df = generator.generate_customer_risk(customer_ids, credit_df, end_date)
        
        num_claims = int(args.num_customers * 0.5)
        claims_df = generator.generate_claims_history(customer_ids, num_claims, end_date)
        
        # For transactions, limit to manageable size or use S3
        if args.use_s3:
            num_transactions = args.num_customers * 50
        else:
            # Limit transactions for direct insert to avoid timeout
            num_transactions = min(args.num_customers * 10, 100000)
        transactions_df = generator.generate_transactions(customer_ids, num_transactions, end_date)
        
        # Load data
        datasets = [
            ("insurance.customer_profiles", profiles_df),
            ("insurance.customer_credit_data", credit_df),
            ("insurance.customer_risk_metrics", risk_df),
            ("insurance.claims_history", claims_df),
            ("insurance.transactions", transactions_df),
        ]
        
        for table_name, df in datasets:
            print(f"\n  Loading {table_name}...")
            if args.use_s3:
                load_dataframe_via_s3(
                    conn, table_name, df,
                    args.s3_bucket, args.s3_prefix, args.s3_iam_role, args.region
                )
            else:
                load_dataframe_direct(conn, table_name, df)
        
        # Step 3: Verify
        verify_data(conn)
        
        print("\n" + "=" * 60)
        print("✓ SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Update feature_store.yaml with your Redshift/S3/DynamoDB settings")
        print("  2. Run: cd ../feature_repo && feast apply")
        print("  3. Run: feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)")
        print("  4. Run: feast serve -p 6566")
        print("=" * 60)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
