#!/usr/bin/env python3
"""
Simplified Server Performance Test - Feature Count Scaling

This script tests the performance of the simplified architecture via the Feast server:
1. Variable feature count latency (5, 10, 20, 40 customer features)
2. ODFV overhead measurement (with/without premium calculator)
3. End-to-end serving performance analysis

The simplified architecture provides focused insights for:
- Batch feature view scaling with feature count
- ODFV transformation overhead
- Real-world serving latency under load
"""

import asyncio
import os
import statistics
import time
from typing import Dict, List, Any
import aiohttp
import json


class SimplifiedServerBenchmark:
    """Benchmark the simplified architecture via Feast server."""

    def __init__(self, base_url: str = None):
        if base_url is None:
            self.base_url = os.getenv('FEAST_SERVER_URL')
            if not self.base_url:
                raise ValueError(
                    "FEAST_SERVER_URL environment variable must be set. "
                    "Example: export FEAST_SERVER_URL='http://localhost:6566'"
                )
        else:
            self.base_url = base_url

        self.base_url = self.base_url.rstrip('/')

        # Test configurations for simplified architecture
        self.test_services = [
            {
                "name": "5 Features Only",
                "service": "customer_5_features",
                "type": "batch_only",
                "feature_count": 5,
                "expected_latency_ms": 15,
                "description": "Minimal customer features"
            },
            {
                "name": "10 Features Only",
                "service": "customer_10_features",
                "type": "batch_only",
                "feature_count": 10,
                "expected_latency_ms": 25,
                "description": "Standard customer features"
            },
            {
                "name": "20 Features Only",
                "service": "customer_20_features",
                "type": "batch_only",
                "feature_count": 20,
                "expected_latency_ms": 40,
                "description": "Comprehensive customer features"
            },
            {
                "name": "40 Features Only",
                "service": "customer_40_features",
                "type": "batch_only",
                "feature_count": 40,
                "expected_latency_ms": 60,
                "description": "All customer features"
            },
            {
                "name": "5 Features + Premium ODFV",
                "service": "customer_5_with_premium",
                "type": "batch_plus_odfv",
                "feature_count": 13,
                "expected_latency_ms": 30,
                "description": "Minimal features + transformation"
            },
            {
                "name": "10 Features + Premium ODFV",
                "service": "customer_10_with_premium",
                "type": "batch_plus_odfv",
                "feature_count": 18,
                "expected_latency_ms": 40,
                "description": "Standard features + transformation"
            },
            {
                "name": "20 Features + Premium ODFV",
                "service": "customer_20_with_premium",
                "type": "batch_plus_odfv",
                "feature_count": 28,
                "expected_latency_ms": 60,
                "description": "Comprehensive features + transformation"
            },
            {
                "name": "40 Features + Premium ODFV",
                "service": "customer_40_with_premium",
                "type": "batch_plus_odfv",
                "feature_count": 48,
                "expected_latency_ms": 80,
                "description": "All features + transformation"
            },
        ]

        # Additional baseline tests
        self.baseline_services = [
            {
                "name": "Customer Baseline",
                "service": "customer_baseline",
                "type": "baseline",
                "feature_count": 40,
                "expected_latency_ms": 50,
                "description": "Pure batch FV (no ODFV)"
            },
            {
                "name": "Premium Only",
                "service": "premium_only",
                "type": "minimal_odfv",
                "feature_count": 11,
                "expected_latency_ms": 25,
                "description": "Minimal ODFV test"
            },
        ]

        self.requests_per_test = 50

    def create_sample_request(self) -> Dict[str, Any]:
        """Create sample request data for all tests."""
        return {
            "entities": {
                "customer_id": ["customer_001"]
            },
            "request_context": {
                "requested_coverage": 250000,
                "requested_deductible": 1000,
                "policy_type": "auto",
                "term_months": 12,
                "additional_drivers": 1,
                "vehicle_age": 5,
            }
        }

    async def make_feature_request(self, session: aiohttp.ClientSession,
                                   service_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single feature request to the Feast server."""
        url = f"{self.base_url}/get-online-features"

        request_payload = {
            "feature_service": service_name,
            **request_data
        }

        start_time = time.perf_counter()
        try:
            async with session.post(url, json=request_payload, timeout=10) as response:
                end_time = time.perf_counter()

                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "latency_ms": (end_time - start_time) * 1000,
                        "response": result,
                        "feature_count": len(result.get("feature_names", [])) if result else 0
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "latency_ms": (end_time - start_time) * 1000,
                        "error": f"HTTP {response.status}: {error_text}",
                        "feature_count": 0
                    }

        except asyncio.TimeoutError:
            end_time = time.perf_counter()
            return {
                "success": False,
                "latency_ms": (end_time - start_time) * 1000,
                "error": "Request timeout",
                "feature_count": 0
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "success": False,
                "latency_ms": (end_time - start_time) * 1000,
                "error": str(e),
                "feature_count": 0
            }

    async def test_service_performance(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test performance for a single service."""
        print(f"🧪 Testing {service_config['name']} ({service_config['service']})...")

        request_data = self.create_sample_request()
        latencies = []
        success_count = 0
        feature_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(self.requests_per_test):
                task = self.make_feature_request(session, service_config['service'], request_data)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and result.get("success"):
                    latencies.append(result["latency_ms"])
                    success_count += 1
                    if feature_count == 0:  # Store feature count from first successful response
                        feature_count = result.get("feature_count", 0)

        return {
            "service": service_config['service'],
            "name": service_config['name'],
            "type": service_config['type'],
            "expected_feature_count": service_config['feature_count'],
            "actual_feature_count": feature_count,
            "expected_latency_ms": service_config['expected_latency_ms'],
            "latencies": latencies,
            "success_count": success_count,
            "total_requests": self.requests_per_test,
            "success_rate": success_count / self.requests_per_test,
        }

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

    def print_service_result(self, result: Dict[str, Any]):
        """Print results for a single service."""
        stats = self.calculate_statistics(result['latencies'])
        if not stats:
            print(f"❌ {result['name']}: No successful requests")
            return

        expected_latency = result['expected_latency_ms']
        actual_latency = stats['mean_ms']
        performance_vs_expected = "✓" if actual_latency <= expected_latency * 1.2 else "⚠️"

        print(f"\n📊 {result['name']}")
        print(f"   Service:           {result['service']}")
        print(f"   Type:              {result['type']}")
        print(f"   Features:          {result['actual_feature_count']} (expected: {result['expected_feature_count']})")
        print(f"   Success Rate:      {result['success_rate']:.1%}")
        print(f"   Mean Latency:      {stats['mean_ms']:.1f} ms (expected: {expected_latency} ms) {performance_vs_expected}")
        print(f"   P95 Latency:       {stats['p95_ms']:.1f} ms")
        print(f"   P99 Latency:       {stats['p99_ms']:.1f} ms")
        print(f"   Std Deviation:     {stats['stdev_ms']:.1f} ms")

    def analyze_scaling_results(self, results: List[Dict[str, Any]]):
        """Analyze feature count scaling patterns."""
        print(f"\n{'='*80}")
        print("📈 FEATURE COUNT SCALING ANALYSIS")
        print(f"{'='*80}")

        batch_only_results = [r for r in results if r.get('type') == 'batch_only']
        batch_plus_odfv_results = [r for r in results if r.get('type') == 'batch_plus_odfv']

        if batch_only_results:
            print(f"\n🔍 BATCH FEATURE VIEW SCALING:")
            print(f"{'Features':>10} {'Mean (ms)':>12} {'P95 (ms)':>12} {'P99 (ms)':>12} {'Efficiency':>12}")
            print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

            for result in batch_only_results:
                stats = self.calculate_statistics(result['latencies'])
                if stats:
                    features = result['actual_feature_count']
                    mean_latency = stats['mean_ms']
                    efficiency = features / mean_latency if mean_latency > 0 else 0

                    print(f"{features:>10} {mean_latency:>12.1f} {stats['p95_ms']:>12.1f} "
                          f"{stats['p99_ms']:>12.1f} {efficiency:>12.1f}")

        if batch_plus_odfv_results:
            print(f"\n🔧 BATCH + ODFV SCALING:")
            print(f"{'Features':>10} {'Mean (ms)':>12} {'P95 (ms)':>12} {'ODFV Overhead':>15}")
            print(f"{'─'*10} {'─'*12} {'─'*12} {'─'*15}")

            for i, result in enumerate(batch_plus_odfv_results):
                stats = self.calculate_statistics(result['latencies'])
                if stats and i < len(batch_only_results):
                    features = result['actual_feature_count']
                    mean_latency = stats['mean_ms']

                    # Calculate ODFV overhead
                    corresponding_batch = batch_only_results[i]
                    batch_stats = self.calculate_statistics(corresponding_batch['latencies'])
                    overhead = mean_latency - batch_stats['mean_ms'] if batch_stats else 0

                    print(f"{features:>10} {mean_latency:>12.1f} {stats['p95_ms']:>12.1f} "
                          f"{overhead:>15.1f}")

    def analyze_odfv_overhead(self, results: List[Dict[str, Any]]):
        """Analyze ODFV transformation overhead."""
        print(f"\n🔧 ODFV OVERHEAD ANALYSIS:")

        baseline_result = next((r for r in results if r.get('service') == 'customer_baseline'), None)
        premium_only_result = next((r for r in results if r.get('service') == 'premium_only'), None)

        if baseline_result and premium_only_result:
            baseline_stats = self.calculate_statistics(baseline_result['latencies'])
            premium_stats = self.calculate_statistics(premium_only_result['latencies'])

            if baseline_stats and premium_stats:
                print(f"   Baseline (40 features, no ODFV):     {baseline_stats['mean_ms']:.1f} ms")
                print(f"   Premium Only (3 + ODFV):            {premium_stats['mean_ms']:.1f} ms")

                # Calculate pure ODFV overhead
                feature_retrieval_per_feature = baseline_stats['mean_ms'] / 40
                estimated_3_feature_latency = feature_retrieval_per_feature * 3
                pure_odfv_overhead = premium_stats['mean_ms'] - estimated_3_feature_latency

                print(f"   Estimated pure ODFV overhead:       {pure_odfv_overhead:.1f} ms")

                if pure_odfv_overhead < 10:
                    print(f"   ✓ Excellent ODFV optimization")
                elif pure_odfv_overhead < 20:
                    print(f"   👍 Good ODFV performance")
                else:
                    print(f"   ⚠️ ODFV overhead may need optimization")

    async def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting Simplified Architecture Server Benchmark")
        print(f"   Server: {self.base_url}")
        print(f"   Requests per service: {self.requests_per_test}")

        all_results = []

        # Test main feature count scaling services
        print(f"\n{'='*80}")
        print("📊 FEATURE COUNT SCALING TESTS")
        print(f"{'='*80}")

        for service_config in self.test_services:
            result = await self.test_service_performance(service_config)
            all_results.append(result)
            self.print_service_result(result)

        # Test baseline services
        print(f"\n{'='*80}")
        print("🎯 BASELINE TESTS")
        print(f"{'='*80}")

        for service_config in self.baseline_services:
            result = await self.test_service_performance(service_config)
            all_results.append(result)
            self.print_service_result(result)

        # Analysis
        self.analyze_scaling_results(all_results)
        self.analyze_odfv_overhead(all_results)

        print(f"\n{'='*80}")
        print("✅ BENCHMARK COMPLETE")
        print(f"{'='*80}")

        # Summary insights
        successful_tests = [r for r in all_results if r['success_count'] > 0]
        if successful_tests:
            all_latencies = []
            for result in successful_tests:
                all_latencies.extend(result['latencies'])

            overall_stats = self.calculate_statistics(all_latencies)
            print(f"\n🎯 OVERALL PERFORMANCE:")
            print(f"   Successful tests:        {len(successful_tests)}/{len(all_results)}")
            print(f"   Overall mean latency:    {overall_stats['mean_ms']:.1f} ms")
            print(f"   Overall P95 latency:     {overall_stats['p95_ms']:.1f} ms")
            print(f"   Overall P99 latency:     {overall_stats['p99_ms']:.1f} ms")

            if overall_stats['p95_ms'] < 100:
                print(f"   🏆 Excellent overall performance!")
            elif overall_stats['p95_ms'] < 200:
                print(f"   👍 Good performance for production use")
            else:
                print(f"   ⚠️ Consider optimization for better user experience")


async def main():
    """Run the simplified server benchmark."""
    try:
        benchmark = SimplifiedServerBenchmark()
        await benchmark.run_full_benchmark()
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())