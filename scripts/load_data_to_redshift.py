#!/usr/bin/env python
"""
Load Sample Data to Redshift

This script loads the generated sample parquet data into Redshift tables
so that Feast can read from RedshiftSource.

There are two methods:
1. Direct insert using redshift_connector (slower, but simpler)
2. Upload to S3 and COPY (faster, recommended for production)

Prerequisites:
- AWS credentials configured
- Redshift cluster running and accessible
- S3 bucket for staging (for COPY method)
- IAM role for Redshift to access S3

Usage:
    # Method 1: Direct insert (smaller datasets)
    python load_data_to_redshift.py --method direct \
        --host your-cluster.xxxxx.redshift.amazonaws.com \
        --database insurance_features \
        --user feast_user \
        --password your_password \
        --data-dir ../data/sample

    # Method 2: S3 COPY (recommended for larger datasets)
    python load_data_to_redshift.py --method s3-copy \
        --host your-cluster.xxxxx.redshift.amazonaws.com \
        --database insurance_features \
        --user feast_user \
        --password your_password \
        --s3-bucket your-feast-bucket \
        --s3-prefix insurance-demo/staging \
        --iam-role arn:aws:iam::123456789012:role/RedshiftS3AccessRole \
        --data-dir ../data/sample

Install dependencies:
    pip install redshift-connector boto3 pandas pyarrow
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

# Try to import redshift_connector
try:
    import redshift_connector
    REDSHIFT_AVAILABLE = True
except ImportError:
    REDSHIFT_AVAILABLE = False
    print("Warning: redshift_connector not installed.")
    print("Install with: pip install redshift-connector")

# Try to import boto3 for S3 operations
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# Table to file mapping
TABLE_FILE_MAPPING = {
    'insurance.customer_profiles': 'customer_profiles.parquet',
    'insurance.customer_credit_data': 'customer_credit.parquet',
    'insurance.customer_risk_metrics': 'customer_risk.parquet',
    'insurance.claims_history': 'claims_history.parquet',
    'insurance.transactions': 'transactions.parquet',
}


def get_redshift_connection(host: str, database: str, user: str, password: str, port: int = 5439):
    """Create a Redshift connection."""
    if not REDSHIFT_AVAILABLE:
        raise ImportError("redshift_connector is required. Install with: pip install redshift-connector")
    
    return redshift_connector.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port,
    )


def load_via_direct_insert(
    host: str,
    database: str,
    user: str,
    password: str,
    data_dir: str,
    port: int = 5439,
    batch_size: int = 1000,
):
    """Load data via direct INSERT statements (slower but simpler)."""
    print("=" * 60)
    print("Loading data via direct INSERT")
    print("=" * 60)
    
    conn = get_redshift_connection(host, database, user, password, port)
    cursor = conn.cursor()
    
    for table_name, file_name in TABLE_FILE_MAPPING.items():
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"  Skipping {table_name}: {file_path} not found")
            continue
        
        print(f"\nLoading {table_name}...")
        df = pd.read_parquet(file_path)
        
        # Convert timestamps to strings for Redshift
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert booleans to proper format
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(str).str.lower()
        
        # Get column names
        columns = list(df.columns)
        column_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        
        # Truncate existing data
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        # Insert in batches
        total_rows = len(df)
        inserted = 0
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]
            values = [tuple(row) for row in batch.values]
            
            insert_sql = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, values)
            
            inserted += len(batch)
            print(f"  Inserted {inserted}/{total_rows} rows", end='\r')
        
        conn.commit()
        print(f"  Inserted {total_rows} rows into {table_name}")
    
    cursor.close()
    conn.close()
    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)


def load_via_s3_copy(
    host: str,
    database: str,
    user: str,
    password: str,
    s3_bucket: str,
    s3_prefix: str,
    iam_role: str,
    data_dir: str,
    region: str = 'us-west-2',
    port: int = 5439,
):
    """Load data via S3 COPY (faster, recommended for production)."""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required for S3 operations. Install with: pip install boto3")
    
    print("=" * 60)
    print("Loading data via S3 COPY")
    print("=" * 60)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region)
    
    # Connect to Redshift
    conn = get_redshift_connection(host, database, user, password, port)
    cursor = conn.cursor()
    
    for table_name, file_name in TABLE_FILE_MAPPING.items():
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"  Skipping {table_name}: {file_path} not found")
            continue
        
        print(f"\nLoading {table_name}...")
        
        # Upload parquet file to S3
        s3_key = f"{s3_prefix}/{file_name}"
        print(f"  Uploading to s3://{s3_bucket}/{s3_key}...")
        s3_client.upload_file(file_path, s3_bucket, s3_key)
        
        # Truncate existing data
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        # COPY from S3
        copy_sql = f"""
        COPY {table_name}
        FROM 's3://{s3_bucket}/{s3_key}'
        IAM_ROLE '{iam_role}'
        FORMAT AS PARQUET;
        """
        
        print(f"  Running COPY command...")
        try:
            cursor.execute(copy_sql)
            conn.commit()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"  Loaded {row_count} rows into {table_name}")
        except Exception as e:
            print(f"  Error loading {table_name}: {e}")
            conn.rollback()
    
    cursor.close()
    conn.close()
    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)


def verify_data(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5439,
):
    """Verify data was loaded correctly."""
    print("\n" + "=" * 60)
    print("Verifying data in Redshift")
    print("=" * 60)
    
    conn = get_redshift_connection(host, database, user, password, port)
    cursor = conn.cursor()
    
    for table_name in TABLE_FILE_MAPPING.keys():
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"  {table_name}: {row_count} rows")
        except Exception as e:
            print(f"  {table_name}: Error - {e}")
    
    cursor.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Load sample data into Redshift for Feast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct insert method
  python load_data_to_redshift.py --method direct \\
      --host my-cluster.xxxxx.us-west-2.redshift.amazonaws.com \\
      --database insurance_features \\
      --user feast_user \\
      --password mypassword

  # S3 COPY method (faster)
  python load_data_to_redshift.py --method s3-copy \\
      --host my-cluster.xxxxx.us-west-2.redshift.amazonaws.com \\
      --database insurance_features \\
      --user feast_user \\
      --password mypassword \\
      --s3-bucket my-feast-bucket \\
      --iam-role arn:aws:iam::123456789012:role/RedshiftS3Role

Environment variables can also be used:
  REDSHIFT_HOST, REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD
        """
    )
    
    parser.add_argument(
        "--method",
        choices=["direct", "s3-copy"],
        default="direct",
        help="Data loading method (default: direct)"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("REDSHIFT_HOST"),
        help="Redshift cluster endpoint (or set REDSHIFT_HOST env var)"
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("REDSHIFT_DATABASE", "insurance_features"),
        help="Redshift database name"
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("REDSHIFT_USER"),
        help="Redshift username (or set REDSHIFT_USER env var)"
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("REDSHIFT_PASSWORD"),
        help="Redshift password (or set REDSHIFT_PASSWORD env var)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5439,
        help="Redshift port (default: 5439)"
    )
    parser.add_argument(
        "--data-dir",
        default="../data/sample",
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 bucket for staging (required for s3-copy method)"
    )
    parser.add_argument(
        "--s3-prefix",
        default="feast-staging",
        help="S3 prefix for staging files"
    )
    parser.add_argument(
        "--iam-role",
        help="IAM role ARN for Redshift to access S3 (required for s3-copy method)"
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing data, don't load"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.host:
        parser.error("--host is required (or set REDSHIFT_HOST env var)")
    if not args.user:
        parser.error("--user is required (or set REDSHIFT_USER env var)")
    if not args.password:
        parser.error("--password is required (or set REDSHIFT_PASSWORD env var)")
    
    if args.method == "s3-copy":
        if not args.s3_bucket:
            parser.error("--s3-bucket is required for s3-copy method")
        if not args.iam_role:
            parser.error("--iam-role is required for s3-copy method")
    
    # Verify only mode
    if args.verify_only:
        verify_data(args.host, args.database, args.user, args.password, args.port)
        return
    
    # Check if data files exist
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Run generate_sample_data.py first to create sample data.")
        sys.exit(1)
    
    # Load data
    if args.method == "direct":
        load_via_direct_insert(
            host=args.host,
            database=args.database,
            user=args.user,
            password=args.password,
            port=args.port,
            data_dir=args.data_dir,
        )
    else:
        load_via_s3_copy(
            host=args.host,
            database=args.database,
            user=args.user,
            password=args.password,
            port=args.port,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            iam_role=args.iam_role,
            region=args.region,
            data_dir=args.data_dir,
        )
    
    # Verify
    verify_data(args.host, args.database, args.user, args.password, args.port)


if __name__ == "__main__":
    main()
