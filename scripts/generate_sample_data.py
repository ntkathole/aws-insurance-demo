#!/usr/bin/env python
"""
Sample Data Generator for AWS Insurance Demo.

This script generates realistic sample data for testing the feature store
without requiring access to production databases.

Usage:
    python generate_sample_data.py --output-dir ../data/sample --num-customers 10000

The generated data can be used for:
- Local development without Redshift
- Testing feature definitions
- Benchmarking online serving latency
- Training example models
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

# Try to import faker for realistic data generation
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("Warning: faker not installed. Using basic random data generation.")
    print("Install with: pip install faker")


class InsuranceDataGenerator:
    """Generate realistic insurance domain sample data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)
        if FAKER_AVAILABLE:
            self.fake = Faker()
            Faker.seed(seed)
        
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
        
        self.policy_types = ['Auto', 'Home', 'Life', 'Health', 'Umbrella']
        
        self.vehicle_makes = [
            'Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes',
            'Nissan', 'Hyundai', 'Kia', 'Subaru', 'Mazda', 'Volkswagen'
        ]
    
    def _generate_timestamps(
        self,
        n: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """Generate random timestamps within a date range."""
        delta = end_date - start_date
        return [
            start_date + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
            for _ in range(n)
        ]
    
    def generate_customer_profiles(
        self,
        num_customers: int,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate customer profile data."""
        print(f"Generating {num_customers} customer profiles...")
        
        customer_ids = [f"CUST{i:08d}" for i in range(1, num_customers + 1)]
        
        # Generate realistic ages with a distribution
        ages = np.clip(np.random.normal(42, 15, num_customers), 18, 85).astype(int)
        
        # Tenure correlates with age
        max_tenure = (ages - 18) * 6  # Max 6 months tenure per year since 18
        tenure_months = [random.randint(1, max(1, mt)) for mt in max_tenure]
        
        data = {
            'customer_id': customer_ids,
            'age': ages,
            'gender': [random.choice(['M', 'F', 'O']) for _ in range(num_customers)],
            'marital_status': [random.choice(['Single', 'Married', 'Divorced', 'Widowed']) 
                             for _ in range(num_customers)],
            'occupation': [random.choice(self.occupations) for _ in range(num_customers)],
            'education_level': [random.choice(self.education_levels) for _ in range(num_customers)],
            'state': [random.choice(self.states) for _ in range(num_customers)],
            'zip_code': [f"{random.randint(10000, 99999)}" for _ in range(num_customers)],
            'urban_rural': [random.choices(['Urban', 'Suburban', 'Rural'], weights=[0.4, 0.4, 0.2])[0]
                          for _ in range(num_customers)],
            'region_risk_zone': [random.randint(1, 5) for _ in range(num_customers)],
            'customer_tenure_months': tenure_months,
            'num_policies': [random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]
                           for _ in range(num_customers)],
            'loyalty_tier': [random.choices(['Bronze', 'Silver', 'Gold', 'Platinum'],
                                           weights=[0.4, 0.35, 0.2, 0.05])[0]
                          for _ in range(num_customers)],
            'has_agent': [random.random() > 0.3 for _ in range(num_customers)],
            'event_timestamp': self._generate_timestamps(num_customers, 
                                                        end_date - timedelta(days=30),
                                                        end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 365)) 
                         for _ in range(num_customers)],
        }
        
        return pd.DataFrame(data)
    
    def generate_customer_credit(
        self,
        customer_ids: List[str],
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate customer credit data."""
        print(f"Generating credit data for {len(customer_ids)} customers...")
        
        num_customers = len(customer_ids)
        
        # Generate credit scores with realistic distribution
        credit_scores = np.clip(np.random.normal(700, 80, num_customers), 300, 850).astype(int)
        
        # Credit tier based on score
        def score_to_tier(score):
            if score >= 800: return 'A'
            elif score >= 740: return 'B'
            elif score >= 670: return 'C'
            elif score >= 580: return 'D'
            else: return 'F'
        
        # Insurance score (similar to credit score but for insurance)
        insurance_scores = np.clip(credit_scores + np.random.normal(0, 30, num_customers), 
                                  300, 900).astype(int)
        
        # Income correlates with credit score
        incomes = np.clip(
            credit_scores * 100 + np.random.normal(20000, 15000, num_customers),
            15000, 500000
        ).astype(int)
        
        data = {
            'customer_id': customer_ids,
            'credit_score': credit_scores,
            'credit_score_tier': [score_to_tier(s) for s in credit_scores],
            'credit_score_change_3m': [random.randint(-30, 30) for _ in range(num_customers)],
            'credit_history_length_months': [random.randint(12, 360) for _ in range(num_customers)],
            'num_credit_accounts': [random.randint(1, 20) for _ in range(num_customers)],
            'num_delinquencies': [random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0]
                                 for _ in range(num_customers)],
            'bankruptcy_flag': [random.random() < 0.02 for _ in range(num_customers)],
            'annual_income': incomes,
            'debt_to_income_ratio': [round(random.uniform(0.1, 0.6), 2) for _ in range(num_customers)],
            'payment_history_score': [round(random.uniform(60, 100), 1) for _ in range(num_customers)],
            'insurance_score': insurance_scores,
            'prior_coverage_lapse': [random.random() < 0.1 for _ in range(num_customers)],
            'event_timestamp': self._generate_timestamps(num_customers, 
                                                        end_date - timedelta(days=7),
                                                        end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 30)) 
                         for _ in range(num_customers)],
        }
        
        return pd.DataFrame(data)
    
    def generate_customer_risk(
        self,
        customer_ids: List[str],
        credit_df: pd.DataFrame,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate customer risk metrics."""
        print(f"Generating risk data for {len(customer_ids)} customers...")
        
        num_customers = len(customer_ids)
        
        # Risk score correlated with credit score
        credit_lookup = dict(zip(credit_df['customer_id'], credit_df['credit_score']))
        
        risk_scores = []
        for cid in customer_ids:
            credit_score = credit_lookup.get(cid, 700)
            # Higher credit = lower risk
            base_risk = 100 - (credit_score - 300) / 5.5
            risk = np.clip(base_risk + np.random.normal(0, 15), 0, 100)
            risk_scores.append(round(risk, 1))
        
        # Claims counts (Poisson distribution)
        claims_1y = np.random.poisson(0.3, num_customers)
        claims_3y = claims_1y + np.random.poisson(0.5, num_customers)
        
        def risk_to_segment(score):
            if score < 40: return 'low'
            elif score < 70: return 'medium'
            else: return 'high'
        
        def risk_to_tier(score):
            if score < 30: return 'preferred'
            elif score < 50: return 'standard'
            elif score < 70: return 'substandard'
            else: return 'decline'
        
        data = {
            'customer_id': customer_ids,
            'overall_risk_score': risk_scores,
            'claims_risk_score': [round(r * random.uniform(0.8, 1.2), 1) for r in risk_scores],
            'fraud_risk_score': [round(random.uniform(0, 30), 1) for _ in range(num_customers)],
            'churn_risk_score': [round(random.uniform(5, 60), 1) for _ in range(num_customers)],
            'num_claims_1y': claims_1y.tolist(),
            'num_claims_3y': claims_3y.tolist(),
            'total_claims_amount_1y': [c * random.uniform(1000, 15000) for c in claims_1y],
            'avg_claim_amount': [random.uniform(500, 10000) for _ in range(num_customers)],
            'policy_changes_1y': [random.randint(0, 3) for _ in range(num_customers)],
            'late_payments_1y': [random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0]
                                for _ in range(num_customers)],
            'inquiry_count_30d': [random.randint(0, 5) for _ in range(num_customers)],
            'driving_violations_3y': [random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]
                                     for _ in range(num_customers)],
            'at_fault_accidents_3y': [random.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0]
                                     for _ in range(num_customers)],
            'dui_flag': [random.random() < 0.01 for _ in range(num_customers)],
            'risk_segment': [risk_to_segment(r) for r in risk_scores],
            'underwriting_tier': [risk_to_tier(r) for r in risk_scores],
            'event_timestamp': self._generate_timestamps(num_customers, 
                                                        end_date - timedelta(days=1),
                                                        end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 7)) 
                         for _ in range(num_customers)],
        }
        
        return pd.DataFrame(data)
    
    def generate_claims_history(
        self,
        customer_ids: List[str],
        num_claims: int,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate claims history data."""
        print(f"Generating {num_claims} claims...")
        
        claim_ids = [f"CLM{i:010d}" for i in range(1, num_claims + 1)]
        
        # Assign claims to customers (some customers have more claims)
        customer_weights = np.random.exponential(1, len(customer_ids))
        customer_weights /= customer_weights.sum()
        claim_customers = np.random.choice(customer_ids, num_claims, p=customer_weights)
        
        # Generate claim amounts (lognormal distribution)
        claim_amounts_requested = np.random.lognormal(8, 1.2, num_claims)
        claim_amounts_requested = np.clip(claim_amounts_requested, 100, 500000)
        
        # Approved amount is usually less than requested
        approval_ratios = np.clip(np.random.beta(8, 2, num_claims), 0, 1)
        claim_amounts_approved = claim_amounts_requested * approval_ratios
        
        # Some claims are still pending
        statuses = ['Approved', 'Denied', 'Pending', 'Under Review', 'Closed']
        status_weights = [0.5, 0.1, 0.1, 0.15, 0.15]
        
        data = {
            'claim_id': claim_ids,
            'customer_id': claim_customers,
            'policy_id': [f"POL{i:08d}" for i in np.random.randint(1, len(customer_ids), num_claims)],
            'provider_id': [f"PROV{i:06d}" for i in np.random.randint(1, 1000, num_claims)],
            'claim_type': [random.choice(self.claim_types) for _ in range(num_claims)],
            'claim_subtype': [f"Subtype_{random.randint(1, 5)}" for _ in range(num_claims)],
            'claim_status': [random.choices(statuses, weights=status_weights)[0] 
                           for _ in range(num_claims)],
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
            'suspicious_indicators_count': [random.choices([0, 1, 2, 3], weights=[0.8, 0.12, 0.05, 0.03])[0]
                                          for _ in range(num_claims)],
            'adjuster_id': [f"ADJ{i:04d}" for i in np.random.randint(1, 100, num_claims)],
            'adjuster_workload': [random.randint(10, 50) for _ in range(num_claims)],
            'priority_score': [round(random.uniform(1, 10), 1) for _ in range(num_claims)],
            'event_timestamp': self._generate_timestamps(num_claims, 
                                                        end_date - timedelta(days=365),
                                                        end_date),
            'created_at': [end_date - timedelta(days=random.randint(1, 365)) 
                         for _ in range(num_claims)],
        }
        
        return pd.DataFrame(data)
    
    def generate_transactions(
        self,
        customer_ids: List[str],
        num_transactions: int,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate transaction data for fraud detection."""
        print(f"Generating {num_transactions} transactions...")
        
        transaction_ids = [f"TXN{i:012d}" for i in range(1, num_transactions + 1)]
        
        # Assign transactions to customers
        customer_weights = np.random.exponential(1, len(customer_ids))
        customer_weights /= customer_weights.sum()
        txn_customers = np.random.choice(customer_ids, num_transactions, p=customer_weights)
        
        # Transaction amounts (lognormal)
        amounts = np.random.lognormal(4, 1.5, num_transactions)
        amounts = np.clip(amounts, 1, 50000)
        
        merchant_categories = [
            'Grocery', 'Gas', 'Restaurant', 'Retail', 'Online', 'Travel',
            'Entertainment', 'Healthcare', 'Utilities', 'Insurance'
        ]
        
        channels = ['Online', 'POS', 'ATM', 'Mobile', 'Phone']
        
        data = {
            'transaction_id': transaction_ids,
            'customer_id': txn_customers,
            'merchant_id': [f"MERCH{i:06d}" for i in np.random.randint(1, 10000, num_transactions)],
            'amount': amounts.round(2).tolist(),
            'currency': ['USD'] * num_transactions,
            'transaction_type': [random.choice(['Purchase', 'Refund', 'Transfer', 'Payment']) 
                               for _ in range(num_transactions)],
            'channel': [random.choices(channels, weights=[0.35, 0.3, 0.1, 0.2, 0.05])[0]
                       for _ in range(num_transactions)],
            'merchant_category': [random.choice(merchant_categories) for _ in range(num_transactions)],
            'merchant_country': [random.choices(['US', 'CA', 'UK', 'Other'], weights=[0.9, 0.04, 0.03, 0.03])[0]
                               for _ in range(num_transactions)],
            'merchant_risk_score': [round(random.uniform(0, 0.5), 2) for _ in range(num_transactions)],
            'device_type': [random.choice(['Mobile', 'Desktop', 'Tablet', 'Unknown']) 
                          for _ in range(num_transactions)],
            'device_fingerprint_match': [random.random() > 0.05 for _ in range(num_transactions)],
            'ip_country': [random.choices(['US', 'CA', 'UK', 'Other'], weights=[0.9, 0.04, 0.03, 0.03])[0]
                         for _ in range(num_transactions)],
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
            'event_timestamp': self._generate_timestamps(num_transactions, 
                                                        end_date - timedelta(days=30),
                                                        end_date),
            'created_at': [end_date - timedelta(days=random.randint(0, 30)) 
                         for _ in range(num_transactions)],
        }
        
        return pd.DataFrame(data)
    
    def generate_all_data(
        self,
        num_customers: int,
        output_dir: str,
        end_date: Optional[datetime] = None
    ):
        """Generate all sample data and save to parquet files."""
        
        if end_date is None:
            end_date = datetime.now()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate customer profiles
        profiles_df = self.generate_customer_profiles(num_customers, end_date)
        profiles_path = os.path.join(output_dir, "customer_profiles.parquet")
        profiles_df.to_parquet(profiles_path, index=False)
        print(f"  Saved: {profiles_path}")
        
        customer_ids = profiles_df['customer_id'].tolist()
        
        # Generate credit data
        credit_df = self.generate_customer_credit(customer_ids, end_date)
        credit_path = os.path.join(output_dir, "customer_credit.parquet")
        credit_df.to_parquet(credit_path, index=False)
        print(f"  Saved: {credit_path}")
        
        # Generate risk data
        risk_df = self.generate_customer_risk(customer_ids, credit_df, end_date)
        risk_path = os.path.join(output_dir, "customer_risk.parquet")
        risk_df.to_parquet(risk_path, index=False)
        print(f"  Saved: {risk_path}")
        
        # Generate claims (roughly 0.5 claims per customer)
        num_claims = int(num_customers * 0.5)
        claims_df = self.generate_claims_history(customer_ids, num_claims, end_date)
        claims_path = os.path.join(output_dir, "claims_history.parquet")
        claims_df.to_parquet(claims_path, index=False)
        print(f"  Saved: {claims_path}")
        
        # Generate transactions (roughly 50 transactions per customer)
        num_transactions = num_customers * 50
        transactions_df = self.generate_transactions(customer_ids, num_transactions, end_date)
        transactions_path = os.path.join(output_dir, "transactions.parquet")
        transactions_df.to_parquet(transactions_path, index=False)
        print(f"  Saved: {transactions_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Customer profiles: {len(profiles_df)} records")
        print(f"Customer credit: {len(credit_df)} records")
        print(f"Customer risk: {len(risk_df)} records")
        print(f"Claims history: {len(claims_df)} records")
        print(f"Transactions: {len(transactions_df)} records")
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
        description="Generate sample data for AWS Insurance Demo"
    )
    parser.add_argument(
        "--output-dir",
        default="../data/sample",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=10000,
        help="Number of customers to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    generator = InsuranceDataGenerator(seed=args.seed)
    generator.generate_all_data(
        num_customers=args.num_customers,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
