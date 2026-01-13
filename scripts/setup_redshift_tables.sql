-- AWS Insurance Demo - Redshift Table Setup
-- 
-- This script creates the necessary tables in Redshift for the insurance demo.
-- Run this script against your Redshift cluster before applying Feast definitions.
--
-- Usage:
--   psql -h your-cluster.xxxxx.us-west-2.redshift.amazonaws.com \
--        -U feast_user -d insurance_features -f setup_redshift_tables.sql

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS insurance;

-- =============================================================================
-- CUSTOMER PROFILE TABLE
-- =============================================================================
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

-- =============================================================================
-- CUSTOMER CREDIT TABLE
-- =============================================================================
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

-- =============================================================================
-- CUSTOMER RISK TABLE
-- =============================================================================
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

-- =============================================================================
-- POLICY TABLE
-- =============================================================================
DROP TABLE IF EXISTS insurance.policy_details;
CREATE TABLE insurance.policy_details (
    policy_id VARCHAR(20) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    policy_type VARCHAR(20),
    product_code VARCHAR(20),
    coverage_amount DECIMAL(15, 2),
    deductible DECIMAL(10, 2),
    premium_monthly DECIMAL(10, 2),
    premium_annual DECIMAL(10, 2),
    policy_start_date VARCHAR(20),
    policy_term_months INT,
    days_until_renewal INT,
    has_collision BOOLEAN,
    has_comprehensive BOOLEAN,
    has_liability BOOLEAN,
    has_uninsured_motorist BOOLEAN,
    has_roadside BOOLEAN,
    multi_policy_discount BOOLEAN,
    safe_driver_discount BOOLEAN,
    paperless_discount BOOLEAN,
    total_discount_pct DECIMAL(5, 2),
    vehicle_year INT,
    vehicle_make VARCHAR(50),
    vehicle_value DECIMAL(15, 2),
    event_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP,
    PRIMARY KEY (policy_id)
)
DISTKEY(customer_id)
SORTKEY(event_timestamp);

-- =============================================================================
-- CLAIMS HISTORY TABLE
-- =============================================================================
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

-- =============================================================================
-- CLAIMS AGGREGATION TABLE (Pre-computed)
-- =============================================================================
DROP TABLE IF EXISTS insurance.claims_aggregations;
CREATE TABLE insurance.claims_aggregations (
    customer_id VARCHAR(20) NOT NULL,
    total_claims_lifetime INT,
    claims_count_1y INT,
    claims_count_3y INT,
    claims_count_5y INT,
    total_claims_amount_lifetime DECIMAL(15, 2),
    total_claims_amount_1y DECIMAL(15, 2),
    total_claims_amount_3y DECIMAL(15, 2),
    avg_claim_amount DECIMAL(15, 2),
    max_claim_amount DECIMAL(15, 2),
    avg_days_between_claims DECIMAL(10, 2),
    days_since_last_claim INT,
    claim_frequency_score DECIMAL(5, 2),
    pct_claims_approved DECIMAL(5, 2),
    pct_claims_denied DECIMAL(5, 2),
    avg_approval_ratio DECIMAL(5, 2),
    fraud_claims_count INT,
    siu_referral_count INT,
    suspicious_claim_ratio DECIMAL(5, 2),
    auto_claims_count INT,
    property_claims_count INT,
    liability_claims_count INT,
    medical_claims_count INT,
    avg_settlement_days DECIMAL(10, 2),
    litigation_rate DECIMAL(5, 2),
    event_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP,
    PRIMARY KEY (customer_id)
)
DISTKEY(customer_id)
SORTKEY(event_timestamp);

-- =============================================================================
-- LAB RESULTS TABLE
-- =============================================================================
DROP TABLE IF EXISTS insurance.lab_results;
CREATE TABLE insurance.lab_results (
    customer_id VARCHAR(20) NOT NULL,
    latest_test_date VARCHAR(20),
    test_provider VARCHAR(100),
    overall_health_score DECIMAL(5, 1),
    cardiovascular_score DECIMAL(5, 1),
    metabolic_score DECIMAL(5, 1),
    liver_function_score DECIMAL(5, 1),
    kidney_function_score DECIMAL(5, 1),
    bmi_category VARCHAR(20),
    smoker_status VARCHAR(20),
    blood_pressure_category VARCHAR(20),
    cholesterol_category VARCHAR(20),
    diabetes_risk_level VARCHAR(20),
    glucose_category VARCHAR(20),
    a1c_category VARCHAR(20),
    ldl_category VARCHAR(20),
    hdl_category VARCHAR(20),
    triglycerides_category VARCHAR(20),
    health_trend_6m VARCHAR(20),
    risk_trend_6m VARCHAR(20),
    num_abnormal_results INT,
    regular_checkups_flag BOOLEAN,
    medication_adherence_score DECIMAL(5, 1),
    event_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP,
    PRIMARY KEY (customer_id)
)
DISTKEY(customer_id)
SORTKEY(event_timestamp);

-- =============================================================================
-- PROVIDER NETWORK TABLE
-- =============================================================================
DROP TABLE IF EXISTS insurance.provider_network;
CREATE TABLE insurance.provider_network (
    provider_id VARCHAR(20) NOT NULL,
    provider_name VARCHAR(200),
    provider_type VARCHAR(50),
    specialty VARCHAR(100),
    network_status VARCHAR(20),
    quality_score DECIMAL(3, 1),
    patient_satisfaction_score DECIMAL(5, 1),
    cost_efficiency_score DECIMAL(5, 1),
    outcome_score DECIMAL(5, 1),
    claims_volume_monthly INT,
    unique_patients_monthly INT,
    avg_claim_amount DECIMAL(15, 2),
    fraud_risk_score DECIMAL(5, 1),
    billing_anomaly_score DECIMAL(5, 1),
    siu_investigation_count INT,
    suspended_flag BOOLEAN,
    referral_count_to INT,
    referral_count_from INT,
    network_centrality_score DECIMAL(5, 2),
    state VARCHAR(2),
    region VARCHAR(50),
    urban_rural VARCHAR(20),
    accredited_flag BOOLEAN,
    years_in_network INT,
    event_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP,
    PRIMARY KEY (provider_id)
)
DISTKEY(provider_id)
SORTKEY(event_timestamp);

-- =============================================================================
-- TRANSACTIONS TABLE
-- =============================================================================
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

-- =============================================================================
-- FEATURE LOGGING TABLE
-- =============================================================================
DROP TABLE IF EXISTS insurance.feature_logs_underwriting;
CREATE TABLE insurance.feature_logs_underwriting (
    log_id VARCHAR(50) NOT NULL,
    feature_service VARCHAR(100),
    entity_key VARCHAR(100),
    request_timestamp TIMESTAMP,
    features_requested TEXT,
    response_time_ms INT,
    PRIMARY KEY (log_id)
)
SORTKEY(request_timestamp);

DROP TABLE IF EXISTS insurance.feature_logs_claims;
CREATE TABLE insurance.feature_logs_claims (
    log_id VARCHAR(50) NOT NULL,
    feature_service VARCHAR(100),
    entity_key VARCHAR(100),
    request_timestamp TIMESTAMP,
    features_requested TEXT,
    response_time_ms INT,
    PRIMARY KEY (log_id)
)
SORTKEY(request_timestamp);

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================
-- Uncomment and modify as needed for your environment
-- GRANT SELECT ON ALL TABLES IN SCHEMA insurance TO feast_user;
-- GRANT INSERT, UPDATE, DELETE ON insurance.feature_logs_underwriting TO feast_user;
-- GRANT INSERT, UPDATE, DELETE ON insurance.feature_logs_claims TO feast_user;

-- =============================================================================
-- SUMMARY
-- =============================================================================
SELECT 
    schemaname, 
    tablename, 
    (SELECT COUNT(*) FROM insurance.customer_profiles) as row_count
FROM pg_tables 
WHERE schemaname = 'insurance'
ORDER BY tablename;

COMMENT ON SCHEMA insurance IS 'Schema for Feast Insurance Demo feature tables';
