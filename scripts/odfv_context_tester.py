#!/usr/bin/env python
"""
ODFV Context Tester - Figure out the correct request context for ODFV services.

This script systematically tests different request context combinations to find
the correct format for ODFV-enabled services like underwriting_quick_quote.
"""

import asyncio
import os
import aiohttp
import json
from typing import Dict, Any, List

class ODFVContextTester:
    """Test different request context combinations for ODFV services."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    async def test_service_with_contexts(self, service_name: str, test_contexts: List[Dict[str, Any]]):
        """Test a service with different context configurations."""
        timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            print(f"🧪 Testing service: {service_name}")

            for i, context in enumerate(test_contexts):
                print(f"\\n  Test {i+1}: {context.get('description', 'No description')}")

                payload = {
                    "feature_service": service_name,
                    "entities": {"customer_id": ["CUST00000001"]},
                    "full_feature_names": False,
                }

                if context.get('request_context'):
                    payload["request_context"] = context['request_context']

                try:
                    async with session.post(
                        f"{self.server_url}/get-online-features",
                        json=payload,
                        ssl=False
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"    ✅ SUCCESS! Features returned: {len(result.get('metadata', {}).get('feature_names', []))}")
                            print(f"       Features: {result.get('metadata', {}).get('feature_names', [])}")
                            return True, context, result
                        else:
                            error_text = await response.text()
                            print(f"    ❌ HTTP {response.status}: {error_text[:200]}")

                except Exception as e:
                    print(f"    ❌ Error: {str(e)[:200]}")

        return False, None, None

    async def test_underwriting_contexts(self):
        """Test various context combinations for underwriting services."""

        # Different context combinations to try
        test_contexts = [
            {
                "description": "No context",
                "request_context": None
            },
            {
                "description": "Basic underwriting context",
                "request_context": {
                    "requested_coverage": 100000,
                    "requested_deductible": 500,
                    "vehicle_age": 3
                }
            },
            {
                "description": "Full underwriting context (standard)",
                "request_context": {
                    "requested_coverage": 100000,
                    "requested_deductible": 500,
                    "policy_type": "auto",
                    "term_months": 12,
                    "additional_drivers": 0,
                    "vehicle_age": 3
                }
            },
            {
                "description": "Alternative context format",
                "request_context": {
                    "coverage": 100000,
                    "deductible": 500,
                    "type": "auto",
                    "term": 12,
                    "drivers": 0,
                    "vehicle_years": 3
                }
            },
            {
                "description": "Minimal premium context",
                "request_context": {
                    "requested_coverage": 100000,
                    "requested_deductible": 500
                }
            }
        ]

        # Test services
        services = ["underwriting_quick_quote", "underwriting_v1", "underwriting_v2"]

        results = {}

        for service in services:
            success, working_context, result = await self.test_service_with_contexts(service, test_contexts)
            results[service] = {
                "success": success,
                "working_context": working_context,
                "result": result
            }

        return results

    async def benchmark_fv_vs_odfv(self, working_contexts: Dict[str, Any]):
        """Once we find working contexts, benchmark FV vs ODFV performance."""

        print("\\n" + "="*60)
        print("📊 BENCHMARKING: Pure FV vs ODFV Performance")
        print("="*60)

        timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

            # Test configurations
            test_configs = [
                {
                    "name": "Pure FV (No ODFVs)",
                    "service": "benchmark_large",
                    "description": "40 features, 3 FVs, 0 ODFVs - Pure DynamoDB reads",
                    "context": None,
                    "expected_operations": "DynamoDB reads only"
                },
                {
                    "name": "Pure FV (Small)",
                    "service": "benchmark_small",
                    "description": "3 features, 1 FV, 0 ODFVs - Minimal DynamoDB reads",
                    "context": None,
                    "expected_operations": "Single DynamoDB read"
                }
            ]

            # Add ODFV tests if we found working contexts
            for service, info in working_contexts.items():
                if info["success"]:
                    odfv_count = 1 if "quick" in service else (1 if "v1" in service else 2)
                    test_configs.append({
                        "name": f"ODFV Service ({service})",
                        "service": service,
                        "description": f"{odfv_count} ODFV{'s' if odfv_count > 1 else ''} - CPU intensive transformations",
                        "context": info["working_context"]["request_context"],
                        "expected_operations": f"DynamoDB reads + {odfv_count} ODFV computation{'s' if odfv_count > 1 else ''}"
                    })

            # Run benchmarks
            benchmark_results = []

            for config in test_configs:
                print(f"\\n🔬 Testing: {config['name']}")
                print(f"   Description: {config['description']}")
                print(f"   Expected: {config['expected_operations']}")

                # Build payload
                payload = {
                    "feature_service": config["service"],
                    "entities": {"customer_id": ["CUST00000001"]},
                    "full_feature_names": False,
                }

                if config["context"]:
                    payload["request_context"] = config["context"]

                # Run multiple requests to get average
                latencies = []
                num_requests = 20

                for i in range(num_requests):
                    try:
                        import time
                        start_time = time.time()

                        async with session.post(
                            f"{self.server_url}/get-online-features",
                            json=payload,
                            ssl=False
                        ) as response:
                            if response.status == 200:
                                await response.json()
                                latency_ms = (time.time() - start_time) * 1000
                                latencies.append(latency_ms)
                            else:
                                print(f"      ⚠️  Request {i+1} failed: HTTP {response.status}")

                    except Exception as e:
                        print(f"      ⚠️  Request {i+1} error: {str(e)[:100]}")

                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)

                    print(f"   ✅ Results: {avg_latency:.1f}ms avg (min: {min_latency:.1f}ms, max: {max_latency:.1f}ms)")

                    benchmark_results.append({
                        "name": config["name"],
                        "service": config["service"],
                        "avg_latency": avg_latency,
                        "min_latency": min_latency,
                        "max_latency": max_latency,
                        "description": config["description"],
                        "expected_operations": config["expected_operations"]
                    })
                else:
                    print(f"   ❌ All requests failed")

            # Analyze results
            self.analyze_fv_vs_odfv_results(benchmark_results)

            return benchmark_results

    def analyze_fv_vs_odfv_results(self, results: List[Dict[str, Any]]):
        """Analyze and display FV vs ODFV performance comparison."""

        print("\\n" + "="*60)
        print("📈 PERFORMANCE ANALYSIS: FV vs ODFV")
        print("="*60)

        # Categorize results
        pure_fv_results = [r for r in results if "Pure FV" in r["name"]]
        odfv_results = [r for r in results if "ODFV" in r["name"]]

        if not pure_fv_results:
            print("❌ No pure FV results to compare")
            return

        if not odfv_results:
            print("❌ No ODFV results to compare")
            return

        # Find baseline (smallest pure FV service)
        baseline = min(pure_fv_results, key=lambda x: x["avg_latency"])
        baseline_latency = baseline["avg_latency"]

        print(f"🏁 Baseline (Pure FV): {baseline['name']} = {baseline_latency:.1f}ms")
        print()

        # Calculate overheads
        print("📊 Performance Comparison:")
        print(f"{'Service':<25} {'Latency':<12} {'Overhead':<12} {'Type':<20}")
        print("-" * 70)

        # Show all results with overhead calculation
        for result in results:
            latency = result["avg_latency"]
            overhead = latency - baseline_latency
            overhead_pct = (overhead / baseline_latency) * 100

            service_type = "Pure FV" if "Pure FV" in result["name"] else "ODFV"

            print(f"{result['service']:<25} {latency:>8.1f}ms {overhead:>+8.1f}ms {service_type:<20}")

        print("-" * 70)

        # Calculate ODFV overhead
        if odfv_results:
            print("\\n🔍 ODFV Overhead Analysis:")

            for odfv in odfv_results:
                overhead = odfv["avg_latency"] - baseline_latency
                overhead_pct = (overhead / baseline_latency) * 100

                # Determine ODFV count from service name
                if "quick" in odfv["service"]:
                    odfv_count = 1
                    odfv_type = "lightweight"
                elif "v1" in odfv["service"]:
                    odfv_count = 1
                    odfv_type = "heavy"
                elif "v2" in odfv["service"]:
                    odfv_count = 2
                    odfv_type = "mixed"
                else:
                    odfv_count = 1
                    odfv_type = "unknown"

                overhead_per_odfv = overhead / odfv_count

                print(f"  • {odfv['service']}: +{overhead:.1f}ms total (+{overhead_pct:.1f}%)")
                print(f"    └─ {odfv_count} {odfv_type} ODFV{'s' if odfv_count > 1 else ''} = {overhead_per_odfv:.1f}ms per ODFV")

        print()
        print("🎯 Key Insights:")
        print("  • Pure FV services = DynamoDB read latency only")
        print("  • ODFV services = DynamoDB reads + CPU computation overhead")
        print("  • ODFV overhead varies by transformation complexity")
        print("  • Use this data for architecture decisions (FV vs ODFV trade-offs)")

async def main():
    """Main function to test ODFV contexts and benchmark performance."""

    # Get server URL from environment variable
    server_url = os.getenv('FEAST_SERVER_URL')
    if not server_url:
        print("❌ Error: FEAST_SERVER_URL environment variable must be set")
        print("   Example: export FEAST_SERVER_URL='https://your-feast-server.com'")
        return

    tester = ODFVContextTester(server_url)

    print("🧪 AWS Insurance Demo - ODFV Context Testing & Performance Comparison")
    print("="*80)

    # Step 1: Find working ODFV contexts
    print("\\n🔍 Step 1: Finding working request contexts for ODFV services...")
    working_contexts = await tester.test_underwriting_contexts()

    # Step 2: Benchmark FV vs ODFV if we found working contexts
    if any(info["success"] for info in working_contexts.values()):
        print("\\n🔬 Step 2: Benchmarking Pure FV vs ODFV performance...")
        benchmark_results = await tester.benchmark_fv_vs_odfv(working_contexts)

        print("\\n✅ Testing complete! Use results to update performance visualizations.")

    else:
        print("\\n❌ No working ODFV contexts found. ODFV services may need:")
        print("   • Different request context format")
        print("   • Proper deployment with all dependencies")
        print("   • Entity data materialization to online store")

        print("\\n💡 Recommendations:")
        print("   • Check ODFV feature view definitions for required context fields")
        print("   • Verify all feature views are properly materialized")
        print("   • Test with different entity values (CUST00000001-50)")

if __name__ == "__main__":
    asyncio.run(main())