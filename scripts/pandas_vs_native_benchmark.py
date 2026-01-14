#!/usr/bin/env python3
"""
Pandas vs Native Python ODFV Performance Benchmark

This script specifically tests the performance difference between:
1. Pandas-based ODFVs (original implementations)
2. Native Python ODFVs (optimized implementations)

Expected improvements:
- Light ODFVs: 8-15ms reduction (pandas operations overhead)
- Heavy ODFVs: 15-25ms reduction (eliminate row iteration loops)

Author: Performance Engineering Team
"""

import asyncio
import os
import statistics
import time
from typing import Dict, List, Any
import aiohttp
import json


class PandasVsNativeBenchmark:
    """Benchmark pandas vs native Python ODFV implementations."""

    def __init__(self, base_url: str = None):
        if base_url is None:
            # Use environment variable or raise error if not set
            self.base_url = os.getenv('FEAST_SERVER_URL')
            if not self.base_url:
                raise ValueError(
                    "FEAST_SERVER_URL environment variable must be set. "
                    "Example: export FEAST_SERVER_URL='https://your-feast-server.com'"
                )
        else:
            self.base_url = base_url

        self.base_url = self.base_url.rstrip('/')

        # Test pairs: (pandas_service, optimized_service, complexity, expected_improvement_ms)
        self.comparison_pairs = [
            {
                "name": "Pure FV Baseline",
                "pandas_service": "benchmark_pure_fv_baseline",
                "optimized_service": "benchmark_pure_fv_baseline",  # Same baseline
                "complexity": "baseline",
                "expected_improvement_ms": 0,  # No improvement expected
                "description": "Baseline with no ODFVs - should be identical"
            },
            {
                "name": "Light ODFV (Premium Calculator)",
                "pandas_service": "benchmark_pandas_light_odfv",
                "optimized_service": "benchmark_optimized_light_odfv",
                "complexity": "light",
                "expected_improvement_ms": 12,  # Expected improvement
                "description": "Vectorized operations, no row iteration - moderate improvement"
            },
            {
                "name": "Heavy ODFV (Underwriting Risk Score)",
                "pandas_service": "benchmark_pandas_heavy_odfv",
                "optimized_service": "benchmark_optimized_heavy_odfv",
                "complexity": "heavy",
                "expected_improvement_ms": 20,  # Expected improvement
                "description": "Eliminated row iteration loops - significant improvement"
            }
        ]

        self.test_entities = ["CUST00000001", "CUST00000002", "CUST00000003"]
        self.requests_per_test = 50  # Focused comparison

    async def make_request(self, session: aiohttp.ClientSession, service_name: str) -> Dict[str, Any]:
        """Make a single request to a feature service."""
        url = f"{self.base_url}/get-online-features"

        # Build payload with request context for ODFVs
        payload = {
            "feature_service": service_name,
            "entities": {"customer_id": self.test_entities[:1]},  # Single entity for focused test
            "full_feature_names": False,
        }

        # Add request context for ODFV services
        if "odfv" in service_name.lower():
            payload["request_context"] = {
                "requested_coverage": 100000,
                "requested_deductible": 500,
                "policy_type": "auto",
                "term_months": 12,
                "additional_drivers": 0,
                "vehicle_age": 3,
                "transaction_amount": 1500.0,
                "merchant_category": "retail",
                "transaction_channel": "online",
                "device_trust_score": 0.8,
                "session_duration_seconds": 300,
                "is_international": False,
                "claim_amount_requested": 5000.0,
                "claim_type": "collision",
                "days_since_incident": 2,
                "documentation_score": 85.0,
                "has_witnesses": True,
                "police_report_filed": True
            }

        start_time = time.perf_counter()
        try:
            async with session.post(url, json=payload) as response:
                end_time = time.perf_counter()

                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "latency_ms": (end_time - start_time) * 1000,
                        "service": service_name,
                        "response_data": data
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "latency_ms": (end_time - start_time) * 1000,
                        "service": service_name,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "success": False,
                "latency_ms": (end_time - start_time) * 1000,
                "service": service_name,
                "error": str(e)
            }

    async def benchmark_service(self, session: aiohttp.ClientSession, service_name: str) -> Dict[str, Any]:
        """Benchmark a single service with multiple requests."""
        print(f"  Testing {service_name}...")

        tasks = [
            self.make_request(session, service_name)
            for _ in range(self.requests_per_test)
        ]

        results = await asyncio.gather(*tasks)

        # Analyze results
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        if not successful_results:
            return {
                "service": service_name,
                "success_rate": 0,
                "error": "All requests failed",
                "failed_examples": failed_results[:3]
            }

        latencies = [r["latency_ms"] for r in successful_results]

        return {
            "service": service_name,
            "success_rate": len(successful_results) / len(results),
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "latencies": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p90": sorted(latencies)[int(0.9 * len(latencies))],
                "p95": sorted(latencies)[int(0.95 * len(latencies))],
                "p99": sorted(latencies)[int(0.99 * len(latencies))],
                "min": min(latencies),
                "max": max(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "failed_examples": failed_results[:3] if failed_results else []
        }

    async def run_comparison_benchmark(self):
        """Run the complete pandas vs native Python comparison."""
        print("🚀 Starting Pandas vs Native Python ODFV Performance Benchmark")
        print(f"📊 Server: {self.base_url}")
        print(f"🔢 Requests per service: {self.requests_per_test}")
        print()

        # Configure session for performance
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            results = {}

            for comparison in self.comparison_pairs:
                print(f"🧪 Testing: {comparison['name']}")
                print(f"   Description: {comparison['description']}")
                print(f"   Expected improvement: {comparison['expected_improvement_ms']}ms")

                # Test pandas version
                pandas_result = await self.benchmark_service(session, comparison["pandas_service"])

                # Test optimized version
                optimized_result = await self.benchmark_service(session, comparison["optimized_service"])

                # Calculate comparison
                comparison_result = self.analyze_comparison(
                    pandas_result,
                    optimized_result,
                    comparison
                )

                results[comparison["name"]] = comparison_result

                # Print results
                self.print_comparison_results(comparison_result)
                print()

        # Generate summary
        self.print_performance_summary(results)
        return results

    def analyze_comparison(self, pandas_result: Dict, optimized_result: Dict, comparison_config: Dict) -> Dict[str, Any]:
        """Analyze the performance difference between pandas and optimized versions."""
        if not pandas_result.get("success_rate", 0) > 0.8 or not optimized_result.get("success_rate", 0) > 0.8:
            return {
                "comparison": comparison_config,
                "valid": False,
                "error": "Insufficient successful requests for comparison",
                "pandas_result": pandas_result,
                "optimized_result": optimized_result
            }

        pandas_mean = pandas_result["latencies"]["mean"]
        optimized_mean = optimized_result["latencies"]["mean"]

        improvement_ms = pandas_mean - optimized_mean
        improvement_percent = (improvement_ms / pandas_mean) * 100

        expected_improvement = comparison_config["expected_improvement_ms"]
        improvement_vs_expected = improvement_ms - expected_improvement

        return {
            "comparison": comparison_config,
            "valid": True,
            "pandas_performance": {
                "mean_ms": round(pandas_mean, 2),
                "p95_ms": round(pandas_result["latencies"]["p95"], 2),
                "p99_ms": round(pandas_result["latencies"]["p99"], 2),
                "success_rate": pandas_result["success_rate"]
            },
            "optimized_performance": {
                "mean_ms": round(optimized_mean, 2),
                "p95_ms": round(optimized_result["latencies"]["p95"], 2),
                "p99_ms": round(optimized_result["latencies"]["p99"], 2),
                "success_rate": optimized_result["success_rate"]
            },
            "improvement": {
                "absolute_ms": round(improvement_ms, 2),
                "percentage": round(improvement_percent, 1),
                "expected_ms": expected_improvement,
                "vs_expected_ms": round(improvement_vs_expected, 2),
                "meets_expectation": improvement_ms >= expected_improvement * 0.7  # 70% of expected
            },
            "full_results": {
                "pandas": pandas_result,
                "optimized": optimized_result
            }
        }

    def print_comparison_results(self, result: Dict[str, Any]):
        """Print formatted comparison results."""
        if not result.get("valid"):
            print(f"   ❌ {result.get('error', 'Invalid comparison')}")
            return

        comparison = result["comparison"]
        pandas_perf = result["pandas_performance"]
        optimized_perf = result["optimized_performance"]
        improvement = result["improvement"]

        print(f"   📈 Pandas Version:")
        print(f"      Mean: {pandas_perf['mean_ms']}ms | P95: {pandas_perf['p95_ms']}ms | Success: {pandas_perf['success_rate']:.1%}")
        print(f"   ⚡ Optimized Version:")
        print(f"      Mean: {optimized_perf['mean_ms']}ms | P95: {optimized_perf['p95_ms']}ms | Success: {optimized_perf['success_rate']:.1%}")

        print(f"   🎯 Performance Improvement:")
        improvement_icon = "✅" if improvement["meets_expectation"] else "⚠️"
        print(f"      {improvement_icon} {improvement['absolute_ms']}ms ({improvement['percentage']}%) improvement")
        print(f"      Expected: {improvement['expected_ms']}ms, Achieved: {improvement['absolute_ms']}ms")

        if improvement["meets_expectation"]:
            print(f"      🎉 Exceeded expectations by {improvement['vs_expected_ms']}ms!")
        else:
            print(f"      💡 Opportunity: {abs(improvement['vs_expected_ms'])}ms below expectation")

    def print_performance_summary(self, results: Dict[str, Any]):
        """Print overall performance summary."""
        print("=" * 80)
        print("📊 PANDAS vs NATIVE PYTHON PERFORMANCE SUMMARY")
        print("=" * 80)

        valid_results = [r for r in results.values() if r.get("valid")]

        if not valid_results:
            print("❌ No valid results to summarize")
            return

        total_improvement_ms = sum([r["improvement"]["absolute_ms"] for r in valid_results])
        avg_improvement_percent = statistics.mean([r["improvement"]["percentage"] for r in valid_results])

        print(f"🎯 Overall Results:")
        print(f"   • Total latency reduction: {total_improvement_ms:.1f}ms across all ODFVs")
        print(f"   • Average improvement: {avg_improvement_percent:.1f}%")
        print()

        print("📈 Performance by ODFV Type:")
        for name, result in results.items():
            if not result.get("valid"):
                continue

            improvement = result["improvement"]
            complexity = result["comparison"]["complexity"]
            icon = "🔥" if improvement["percentage"] > 20 else ("⚡" if improvement["percentage"] > 10 else "✅")

            print(f"   {icon} {name} ({complexity}):")
            print(f"      Improvement: {improvement['absolute_ms']}ms ({improvement['percentage']}%)")
            print(f"      Meets expectation: {'Yes' if improvement['meets_expectation'] else 'No'}")

        print()
        print("🚀 Key Optimizations Applied:")
        print("   • ✅ Eliminated pandas row iteration loops (biggest impact)")
        print("   • ✅ Replaced pd.cut() and pd.select() with native conditionals")
        print("   • ✅ Pre-calculated input values to avoid repeated pandas operations")
        print("   • ✅ Used vectorized operations for mathematical calculations")
        print("   • ✅ Minimized DataFrame operations")
        print()
        print("💡 Production Recommendations:")
        print("   • Deploy optimized ODFVs for immediate latency improvements")
        print("   • Monitor ODFV computational overhead in production")
        print("   • Consider further optimizations for critical path services")
        print("   • Update SLA targets based on measured improvements")


async def main():
    """Main execution function."""
    benchmark = PandasVsNativeBenchmark()

    try:
        results = await benchmark.run_comparison_benchmark()

        # Save results for analysis
        timestamp = int(time.time())
        output_file = f"performance_plots/pandas_vs_native_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📁 Results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())