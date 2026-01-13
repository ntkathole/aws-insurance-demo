#!/usr/bin/env python
"""
FV vs ODFV Performance Benchmark - The Critical Missing Dimension

This script demonstrates the performance difference between:
1. Pure Feature Views (FV) - Just DynamoDB reads
2. On-Demand Feature Views (ODFV) - DynamoDB reads + CPU computations

This is one of the most important performance factors in Feast.
"""

import asyncio
import os
import aiohttp
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class FVvsODFVBenchmark:
    """Benchmark Pure FV vs ODFV performance."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    async def benchmark_performance(self, num_requests: int = 100):
        """Benchmark both FV and ODFV performance."""

        timeout = aiohttp.ClientTimeout(total=60.0, connect=10.0)
        connector = aiohttp.TCPConnector(ssl=False, limit=100, limit_per_host=10)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

            # Test configurations
            configs = [
                {
                    "name": "Pure FV - Small (3 features)",
                    "type": "pure_fv",
                    "service": "benchmark_small",
                    "description": "3 features from 1 feature view - minimal DynamoDB read",
                    "expected_operations": "Single DynamoDB read",
                    "payload": {
                        "feature_service": "benchmark_small",
                        "entities": {"customer_id": ["CUST00000001"]},
                        "full_feature_names": False,
                    }
                },
                {
                    "name": "Pure FV - Medium (12 features)",
                    "type": "pure_fv",
                    "service": "benchmark_medium",
                    "description": "12 features from 3 feature views - multiple DynamoDB reads",
                    "expected_operations": "3 DynamoDB reads + joins",
                    "payload": {
                        "feature_service": "benchmark_medium",
                        "entities": {"customer_id": ["CUST00000001"]},
                        "full_feature_names": False,
                    }
                },
                {
                    "name": "Pure FV - Large (40 features)",
                    "type": "pure_fv",
                    "service": "benchmark_large",
                    "description": "40 features from 3 feature views - multiple DynamoDB reads",
                    "expected_operations": "3 DynamoDB reads + joins",
                    "payload": {
                        "feature_service": "benchmark_large",
                        "entities": {"customer_id": ["CUST00000001"]},
                        "full_feature_names": False,
                    }
                }
            ]

            # Try to add ODFV configs (may fail if not properly deployed)
            odfv_configs = [
                {
                    "name": "ODFV - Lightweight (premium_calculator)",
                    "type": "odfv_light",
                    "service": "underwriting_quick_quote",
                    "description": "Lightweight ODFV - vectorized operations",
                    "expected_operations": "3 DynamoDB reads + lightweight CPU computation",
                    "payload": {
                        "feature_service": "underwriting_quick_quote",
                        "entities": {
                            "customer_id": ["CUST00000001"],
                            # Try providing request data as entities
                            "requested_coverage": [100000],
                            "requested_deductible": [500],
                            "policy_type": ["auto"],
                            "term_months": [12],
                            "additional_drivers": [0],
                            "vehicle_age": [3]
                        },
                        "full_feature_names": False,
                    }
                },
                {
                    "name": "ODFV - Heavy (underwriting_risk_score)",
                    "type": "odfv_heavy",
                    "service": "underwriting_v1",
                    "description": "Heavy ODFV - row iteration + complex logic",
                    "expected_operations": "3 DynamoDB reads + heavy CPU computation",
                    "payload": {
                        "feature_service": "underwriting_v1",
                        "entities": {
                            "customer_id": ["CUST00000001"],
                            "requested_coverage": [100000],
                            "requested_deductible": [500],
                            "policy_type": ["auto"],
                            "term_months": [12],
                            "additional_drivers": [0],
                            "vehicle_age": [3]
                        },
                        "full_feature_names": False,
                    }
                }
            ]

            print("🔬 FV vs ODFV Performance Benchmark")
            print("="*60)

            results = []

            # Test Pure FV services (these should work)
            for config in configs:
                print(f"\\n📊 Testing: {config['name']}")
                print(f"   Description: {config['description']}")

                latencies = await self._run_requests(session, config['payload'], num_requests)

                if latencies:
                    avg_lat = statistics.mean(latencies)
                    p99_lat = self._percentile(latencies, 99)

                    print(f"   ✅ {avg_lat:.1f}ms avg, {p99_lat:.1f}ms p99 ({len(latencies)}/{num_requests} success)")

                    results.append({
                        **config,
                        "latencies": latencies,
                        "avg_latency": avg_lat,
                        "p99_latency": p99_lat,
                        "success": True
                    })
                else:
                    print(f"   ❌ Failed")
                    results.append({**config, "success": False})

            # Test ODFV services (may fail due to context issues)
            print(f"\\n🧪 Testing ODFV Services (may fail due to deployment/context):")

            for config in odfv_configs:
                print(f"\\n📊 Testing: {config['name']}")
                print(f"   Description: {config['description']}")

                latencies = await self._run_requests(session, config['payload'], num_requests)

                if latencies:
                    avg_lat = statistics.mean(latencies)
                    p99_lat = self._percentile(latencies, 99)

                    print(f"   ✅ {avg_lat:.1f}ms avg, {p99_lat:.1f}ms p99 ({len(latencies)}/{num_requests} success)")

                    results.append({
                        **config,
                        "latencies": latencies,
                        "avg_latency": avg_lat,
                        "p99_latency": p99_lat,
                        "success": True
                    })
                else:
                    print(f"   ❌ Failed - ODFV context/deployment issue")
                    results.append({**config, "success": False})

            return results

    async def _run_requests(self, session: aiohttp.ClientSession, payload: Dict[str, Any], num_requests: int) -> List[float]:
        """Run multiple requests and return latencies."""
        latencies = []

        for i in range(num_requests):
            try:
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

            except Exception:
                pass  # Ignore errors, just collect successful requests

        return latencies

    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((p / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and display FV vs ODFV comparison."""

        print("\\n" + "="*60)
        print("📈 FV vs ODFV PERFORMANCE ANALYSIS")
        print("="*60)

        # Separate results
        fv_results = [r for r in results if r.get('success') and r['type'] == 'pure_fv']
        odfv_results = [r for r in results if r.get('success') and r['type'].startswith('odfv')]

        if not fv_results:
            print("❌ No successful FV results")
            return

        # Find baseline
        baseline = min(fv_results, key=lambda x: x['avg_latency'])
        baseline_latency = baseline['avg_latency']

        print(f"🏁 Baseline (Minimal FV): {baseline['name']} = {baseline_latency:.1f}ms")
        print()

        # Performance comparison table
        print("📊 Performance Comparison:")
        print(f"{'Test Name':<40} {'Avg (ms)':<10} {'P99 (ms)':<10} {'Overhead':<15} {'Type':<15}")
        print("-" * 95)

        all_successful = fv_results + odfv_results

        for result in all_successful:
            overhead = result['avg_latency'] - baseline_latency
            overhead_str = f"+{overhead:.1f}ms"
            type_str = "Pure FV" if result['type'] == 'pure_fv' else "ODFV"

            print(f"{result['name']:<40} {result['avg_latency']:<10.1f} {result['p99_latency']:<10.1f} {overhead_str:<15} {type_str:<15}")

        # Analysis
        print("\\n🔍 KEY INSIGHTS:")

        if len(fv_results) > 1:
            feature_scaling = []
            for r in fv_results:
                if 'Small' in r['name']:
                    features = 3
                elif 'Medium' in r['name']:
                    features = 12
                elif 'Large' in r['name']:
                    features = 40
                else:
                    features = 0

                if features > 0:
                    feature_scaling.append((features, r['avg_latency']))

            if len(feature_scaling) > 1:
                feature_scaling.sort()
                slope = (feature_scaling[-1][1] - feature_scaling[0][1]) / (feature_scaling[-1][0] - feature_scaling[0][0])
                print(f"  • Feature Scaling: ~{slope:.2f}ms per additional feature (Pure FV)")

        if odfv_results:
            print(f"  • ODFV Overhead: Found {len(odfv_results)} working ODFV service(s)")
            for odfv in odfv_results:
                overhead = odfv['avg_latency'] - baseline_latency
                overhead_pct = (overhead / baseline_latency) * 100
                complexity = "Lightweight" if "light" in odfv['type'] else "Heavy"
                print(f"    └─ {complexity} ODFV: +{overhead:.1f}ms (+{overhead_pct:.1f}%) overhead")
        else:
            print(f"  • ODFV Services: Not tested due to deployment/context issues")

        print(f"\\n  • Pure FV = DynamoDB read latency only")
        print(f"  • ODFV = DynamoDB reads + CPU computation overhead")
        print(f"  • Use this data for architectural decisions (when to use ODFVs)")

        return all_successful

    def create_fv_vs_odfv_plot(self, results: List[Dict[str, Any]]):
        """Create visualization comparing FV vs ODFV performance."""

        if not results:
            print("❌ No results to plot")
            return

        # Prepare data
        names = [r['name'].replace(' (', '\\n(').replace(')', ')') for r in results]
        avg_latencies = [r['avg_latency'] for r in results]
        types = [r['type'] for r in results]

        # Create colors
        colors = []
        for t in types:
            if t == 'pure_fv':
                colors.append('#4ECDC4')
            elif t == 'odfv_light':
                colors.append('#FFA07A')
            elif t == 'odfv_heavy':
                colors.append('#FF6B6B')
            else:
                colors.append('#45B7D1')

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Bar chart comparison
        bars = ax1.bar(range(len(names)), avg_latencies, color=colors, alpha=0.8)
        ax1.set_xlabel('Service Type')
        ax1.set_ylabel('Average Latency (ms)')
        ax1.set_title('FV vs ODFV Performance Comparison\\n(The Critical Difference)', fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, avg_latencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')

        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='Pure FV (DynamoDB only)'),
            plt.Rectangle((0,0),1,1, facecolor='#FFA07A', alpha=0.8, label='Lightweight ODFV'),
            plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='Heavy ODFV')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')

        # Plot 2: Overhead analysis
        if len(results) > 0:
            baseline = min([r['avg_latency'] for r in results if r['type'] == 'pure_fv'], default=results[0]['avg_latency'])
            overheads = [r['avg_latency'] - baseline for r in results]
            overhead_pcts = [(oh / baseline) * 100 for oh in overheads]

            bars2 = ax2.bar(range(len(names)), overhead_pcts, color=colors, alpha=0.8)
            ax2.set_xlabel('Service Type')
            ax2.set_ylabel('Overhead (%)')
            ax2.set_title('ODFV Computational Overhead\\n(% increase over baseline)', fontweight='bold')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars2, overhead_pcts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.suptitle('Pure Feature Views vs On-Demand Feature Views\\nCPU vs I/O Performance Comparison',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('performance_plots/06_fv_vs_odfv_comparison.jpg', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created: 06_fv_vs_odfv_comparison.jpg")


async def main():
    """Main function."""

    # Get server URL from environment variable
    server_url = os.getenv('FEAST_SERVER_URL')
    if not server_url:
        print("❌ Error: FEAST_SERVER_URL environment variable must be set")
        print("   Example: export FEAST_SERVER_URL='https://your-feast-server.com'")
        return

    benchmark = FVvsODFVBenchmark(server_url)

    print("🚀 FV vs ODFV Performance Benchmark - AWS Insurance Demo")
    print("="*80)
    print("This test measures the critical difference between:")
    print("  • Pure FVs: DynamoDB reads only")
    print("  • ODFVs: DynamoDB reads + CPU computations")
    print()

    # Run benchmark
    results = await benchmark.benchmark_performance(num_requests=100)

    # Analyze results
    successful_results = benchmark.analyze_results(results)

    # Create visualization if we have results
    if successful_results:
        benchmark.create_fv_vs_odfv_plot(successful_results)

        print("\\n✅ FV vs ODFV analysis complete!")
        print("📂 Check performance_plots/06_fv_vs_odfv_comparison.jpg")

        # Provide architectural guidance
        print("\\n🎯 ARCHITECTURAL GUIDANCE:")
        print("  • Use Pure FVs when: Simple aggregations, lookups, historical data")
        print("  • Use ODFVs when: Real-time calculations, business logic, transformations")
        print("  • ODFV overhead: Budget 10-50ms per ODFV depending on complexity")
        print("  • Critical decision: I/O bound (FV) vs CPU bound (ODFV) operations")

    else:
        print("\\n⚠️  Limited results - ODFV services need proper deployment/context")
        print("💡 Still valuable: Shows pure FV performance characteristics")


if __name__ == "__main__":
    asyncio.run(main())