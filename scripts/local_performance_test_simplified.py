#!/usr/bin/env python3
"""
Simplified Local Performance Test - Premium Calculator Only

This script tests the performance of the premium calculator ODFV
in the simplified architecture. It focuses on:

1. Premium calculator transformation latency (single ODFV)
2. Feature count scaling simulation (5, 10, 20, 40 features)
3. Computational overhead measurement

Since we simplified to only one ODFV, this provides focused performance insights
for the premium calculator optimization.
"""

import time
import statistics
from typing import Dict, Any, List
import sys
import os

# Add feature_repo to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_repo'))

# Import the simplified premium calculator function directly
try:
    # Import the transformation function directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "on_demand_features",
        os.path.join(os.path.dirname(__file__), '..', 'feature_repo', 'on_demand_features.py')
    )
    odfv_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(odfv_module)

    # Get the transformation function
    premium_calculator_optimized = odfv_module.premium_calculator_optimized._func
    print("✓ Loaded simplified premium calculator transformation function")
except Exception as e:
    print(f"❌ Failed to import simplified ODFV: {e}")

    # Fallback: create a test function
    def premium_calculator_optimized(input_dict):
        """Test implementation of premium calculator for performance testing."""
        import math

        def safe_get(key, default):
            value = input_dict.get(key)
            if value is None:
                return default
            if isinstance(value, list):
                return value[0] if value else default
            if isinstance(value, float) and math.isnan(value):
                return default
            return value

        age = safe_get("age", 35)
        region_risk_zone = safe_get("region_risk_zone", 3)
        vehicle_age = safe_get("vehicle_age", 3)
        coverage_amount = safe_get("requested_coverage", 100000)
        deductible = safe_get("requested_deductible", 500)
        credit_score = safe_get("credit_score", 700)

        # Age factor calculation
        if age < 25:
            age_factor = 1.4
        elif age < 30:
            age_factor = 1.1
        elif age <= 65:
            age_factor = 1.0
        elif age <= 75:
            age_factor = 1.2
        else:
            age_factor = 1.5

        # Location factor
        location_map = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.35}
        location_factor = location_map.get(region_risk_zone, 1.0)

        # Vehicle age factor
        if vehicle_age <= 2:
            vehicle_age_factor = 1.15
        elif vehicle_age <= 5:
            vehicle_age_factor = 1.0
        elif vehicle_age <= 10:
            vehicle_age_factor = 0.9
        else:
            vehicle_age_factor = 0.85

        # Coverage and deductible calculations
        coverage_factor = max(0.8, min(1.4, 0.8 + (coverage_amount / 500000) * 0.4))
        deductible_credit = max(0.7, min(1.0, 1.0 - (deductible / 5000)))

        # Base premium calculation
        base_rate = 800
        estimated_base_premium = round(
            base_rate * age_factor * location_factor *
            vehicle_age_factor * coverage_factor * deductible_credit,
            2
        )

        # Credit adjustment
        if credit_score >= 800:
            credit_adjustment = 0.85
        elif credit_score >= 700:
            credit_adjustment = 1.0
        elif credit_score >= 600:
            credit_adjustment = 1.15
        else:
            credit_adjustment = 1.35

        estimated_annual_premium = round(estimated_base_premium * credit_adjustment, 2)
        estimated_monthly_premium = round(estimated_annual_premium / 12, 2)

        return {
            "age_factor": float(age_factor),
            "location_factor": float(location_factor),
            "vehicle_age_factor": float(vehicle_age_factor),
            "coverage_factor": float(coverage_factor),
            "deductible_credit": float(deductible_credit),
            "estimated_base_premium": float(estimated_base_premium),
            "estimated_monthly_premium": float(estimated_monthly_premium),
            "estimated_annual_premium": float(estimated_annual_premium),
        }

    print("✓ Using fallback premium calculator implementation")


class SimplifiedPerformanceTest:
    """Test premium calculator performance in simplified architecture."""

    def __init__(self):
        self.iterations = 1000
        self.feature_counts = [5, 10, 20, 40]

    def create_customer_sample_data(self, feature_count: int = 40) -> Dict[str, Any]:
        """Create sample customer data with variable feature counts."""

        # Base customer data for premium calculator
        base_data = {
            # Essential features for premium calculator
            "age": 35,
            "region_risk_zone": 3,
            "credit_score": 720,

            # Request-time data
            "requested_coverage": 250000,
            "requested_deductible": 1000,
            "policy_type": "auto",
            "term_months": 12,
            "additional_drivers": 1,
            "vehicle_age": 5,
        }

        # Extended customer features for feature count testing
        extended_features = {
            # Profile features (13 total)
            "gender": "M",
            "marital_status": "married",
            "occupation": "engineer",
            "education_level": "bachelors",
            "state": "CA",
            "zip_code": "90210",
            "urban_rural": "urban",
            "customer_tenure_months": 36,
            "num_policies": 2,
            "loyalty_tier": "gold",
            "has_agent": True,

            # Credit features (11 total)
            "credit_score_tier": "A",
            "credit_score_change_3m": 15,
            "credit_history_length_months": 120,
            "num_credit_accounts": 8,
            "num_delinquencies": 0,
            "bankruptcy_flag": False,
            "annual_income": 85000.0,
            "debt_to_income_ratio": 0.35,
            "payment_history_score": 92.5,
            "insurance_score": 780,
            "prior_coverage_lapse": False,

            # Risk features (16 total)
            "overall_risk_score": 25.5,
            "claims_risk_score": 15.2,
            "fraud_risk_score": 8.1,
            "churn_risk_score": 12.3,
            "num_claims_1y": 0,
            "num_claims_3y": 1,
            "total_claims_amount_1y": 0.0,
            "avg_claim_amount": 2500.0,
            "policy_changes_1y": 0,
            "late_payments_1y": 0,
            "inquiry_count_30d": 1,
            "driving_violations_3y": 0,
            "at_fault_accidents_3y": 0,
            "dui_flag": False,
            "risk_segment": "low",
            "underwriting_tier": "preferred",
        }

        # Combine features based on desired count
        if feature_count <= 5:
            # Just essential features for premium calculator
            return base_data
        elif feature_count <= 10:
            result = base_data.copy()
            # Add first 10 profile features
            profile_keys = list(extended_features.keys())[:feature_count-6]  # -6 for base features
            for key in profile_keys:
                result[key] = extended_features[key]
            return result
        elif feature_count <= 20:
            result = base_data.copy()
            # Add first 20 features
            all_keys = list(extended_features.keys())[:feature_count-6]
            for key in all_keys:
                result[key] = extended_features[key]
            return result
        else:
            # All features
            result = base_data.copy()
            result.update(extended_features)
            return result

    def test_premium_calculator_performance(self) -> Dict[str, List[float]]:
        """Test premium calculator performance."""
        print(f"🧪 Testing premium calculator performance ({self.iterations} iterations)...")

        # Create sample data
        sample_data = self.create_customer_sample_data()

        # Test performance
        latencies = []
        successful_runs = 0

        for i in range(self.iterations):
            try:
                start_time = time.perf_counter()
                result = premium_calculator_optimized(sample_data)
                end_time = time.perf_counter()

                # Validate result
                if isinstance(result, dict) and "estimated_annual_premium" in result:
                    latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    successful_runs += 1

            except Exception as e:
                print(f"❌ Error in iteration {i}: {e}")
                continue

        print(f"✓ Completed {successful_runs}/{self.iterations} successful runs")
        return {"premium_calculator": latencies}

    def test_feature_count_scaling(self) -> Dict[str, Dict[str, List[float]]]:
        """Test how latency scales with feature count."""
        print(f"\n🔍 Testing feature count scaling...")

        results = {}

        for feature_count in self.feature_counts:
            print(f"  Testing {feature_count} features...")
            sample_data = self.create_customer_sample_data(feature_count)

            latencies = []
            for i in range(min(500, self.iterations)):  # Fewer iterations for scaling test
                try:
                    start_time = time.perf_counter()
                    result = premium_calculator_optimized(sample_data)
                    end_time = time.perf_counter()

                    if isinstance(result, dict) and "estimated_annual_premium" in result:
                        latencies.append((end_time - start_time) * 1000)

                except Exception:
                    continue

            results[f"{feature_count}_features"] = {
                "premium_calculator": latencies
            }

        return results

    def calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate performance statistics."""
        if not latencies:
            return {}

        return {
            "count": len(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "p99_ms": sorted(latencies)[int(0.99 * len(latencies))],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

    def print_results(self, results: Dict[str, List[float]]):
        """Print performance results in a formatted table."""
        print(f"\n{'='*80}")
        print("🏆 PREMIUM CALCULATOR PERFORMANCE RESULTS")
        print(f"{'='*80}")

        for test_name, latencies in results.items():
            stats = self.calculate_statistics(latencies)
            if not stats:
                continue

            print(f"\n📊 {test_name.upper().replace('_', ' ')}")
            print(f"{'─'*50}")
            print(f"Successful runs: {stats['count']:,}")
            print(f"Mean latency:    {stats['mean_ms']:.2f} ms")
            print(f"Median latency:  {stats['median_ms']:.2f} ms")
            print(f"P95 latency:     {stats['p95_ms']:.2f} ms")
            print(f"P99 latency:     {stats['p99_ms']:.2f} ms")
            print(f"Min latency:     {stats['min_ms']:.2f} ms")
            print(f"Max latency:     {stats['max_ms']:.2f} ms")
            print(f"Std deviation:   {stats['stdev_ms']:.2f} ms")

    def print_scaling_results(self, results: Dict[str, Dict[str, List[float]]]):
        """Print feature count scaling results."""
        print(f"\n{'='*80}")
        print("📈 FEATURE COUNT SCALING ANALYSIS")
        print(f"{'='*80}")

        print(f"{'Features':>10} {'Mean (ms)':>12} {'P95 (ms)':>12} {'P99 (ms)':>12} {'Std Dev':>12}")
        print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

        for feature_count in self.feature_counts:
            key = f"{feature_count}_features"
            if key in results and "premium_calculator" in results[key]:
                stats = self.calculate_statistics(results[key]["premium_calculator"])
                if stats:
                    print(f"{feature_count:>10} {stats['mean_ms']:>12.2f} {stats['p95_ms']:>12.2f} "
                          f"{stats['p99_ms']:>12.2f} {stats['stdev_ms']:>12.2f}")

    def run_full_test(self):
        """Run the complete performance test suite."""
        print("🚀 Starting Simplified Architecture Performance Test")
        print(f"   Testing premium calculator ODFV only")
        print(f"   {self.iterations:,} iterations per test")

        # Test 1: Basic premium calculator performance
        basic_results = self.test_premium_calculator_performance()
        self.print_results(basic_results)

        # Test 2: Feature count scaling
        scaling_results = self.test_feature_count_scaling()
        self.print_scaling_results(scaling_results)

        print(f"\n{'='*80}")
        print("✅ PERFORMANCE TEST COMPLETE")
        print(f"{'='*80}")

        # Summary insights
        if "premium_calculator" in basic_results:
            basic_stats = self.calculate_statistics(basic_results["premium_calculator"])
            if basic_stats:
                print(f"\n🎯 KEY INSIGHTS:")
                print(f"   • Premium calculator mean latency: {basic_stats['mean_ms']:.2f} ms")
                print(f"   • P95 latency target (<15ms):      {'✓' if basic_stats['p95_ms'] < 15 else '❌'}")
                print(f"   • Consistency (low std dev):       {'✓' if basic_stats['stdev_ms'] < 2 else '❌'}")

                if basic_stats['mean_ms'] < 10:
                    print(f"   🏆 Excellent performance! Well optimized for production.")
                elif basic_stats['mean_ms'] < 20:
                    print(f"   👍 Good performance, suitable for real-time serving.")
                else:
                    print(f"   ⚠️  Consider further optimization for better latency.")


def main():
    """Run the simplified performance test."""
    try:
        tester = SimplifiedPerformanceTest()
        tester.run_full_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()