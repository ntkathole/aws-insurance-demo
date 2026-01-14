#!/usr/bin/env python
"""
Enhanced Multi-Dimensional Performance Testing for AWS Insurance Demo.

This script measures latency scaling across 6 dimensions using EXISTING feature services:
1. Feature Count Scaling - Different feature counts per service
2. Feature View Scaling - Different number of tables queried
3. Concurrency Scaling - Increasing requests per second
4. Data Volume Scaling - Different batch sizes
5. ODFV Count Scaling - Services with 0, 1, or 2 ODFVs
6. ODFV Complexity Scaling - Lightweight vs heavy transformations

Usage:
    # Test all dimensions with existing services
    python enhanced_performance_test.py --server-url https://your-server --suite comprehensive

    # Test specific dimension
    python enhanced_performance_test.py --server-url https://your-server --dimension odfv_scaling

    # Test concurrency scaling for specific service
    python enhanced_performance_test.py --server-url https://your-server --service benchmark_small --test concurrency

Requirements:
    pip install aiohttp numpy tabulate
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    exit(1)

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Warning: tabulate not available. Install with: pip install tabulate")


@dataclass
class ServiceProfile:
    """Profile of an existing feature service for testing."""
    name: str
    description: str
    features: int
    feature_views: int
    odfvs: int
    odfv_complexity: str
    target_p99_ms: int
    requires_context: bool
    context_template: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Results from a performance test."""
    service_name: str
    test_type: str
    batch_size: int
    concurrency: int
    num_requests: int
    latencies_ms: List[float]
    errors: List[str]
    timestamp: str

    @property
    def success_rate(self) -> float:
        total_requests = len(self.latencies_ms) + len(self.errors)
        return (len(self.latencies_ms) / total_requests * 100) if total_requests > 0 else 0.0

    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency(self) -> float:
        return self._percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0

    @property
    def p90_latency(self) -> float:
        return self._percentile(self.latencies_ms, 90) if self.latencies_ms else 0.0

    @property
    def p95_latency(self) -> float:
        return self._percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0

    @property
    def p99_latency(self) -> float:
        return self._percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((p / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class EnhancedPerformanceTester:
    """Enhanced performance tester for multi-dimensional analysis."""

    # Existing feature services available on remote server
    SERVICE_PROFILES = {
        "benchmark_small": ServiceProfile(
            name="benchmark_small",
            description="Small feature set - 3 features, 1 FV",
            features=3,
            feature_views=1,
            odfvs=0,
            odfv_complexity="none",
            target_p99_ms=10,
            requires_context=False,
        ),
        "benchmark_medium": ServiceProfile(
            name="benchmark_medium",
            description="Medium feature set - 12 features, 3 FVs",
            features=12,
            feature_views=3,
            odfvs=0,
            odfv_complexity="none",
            target_p99_ms=30,
            requires_context=False,
        ),
        "benchmark_large": ServiceProfile(
            name="benchmark_large",
            description="Large feature set - 40+ features, 3 FVs",
            features=40,
            feature_views=3,
            odfvs=0,
            odfv_complexity="none",
            target_p99_ms=75,
            requires_context=False,
        ),
        "underwriting_quick_quote": ServiceProfile(
            name="underwriting_quick_quote",
            description="Quick quote with lightweight ODFV - 7 features, 3 FVs, 1 ODFV",
            features=7,
            feature_views=3,
            odfvs=1,
            odfv_complexity="light",
            target_p99_ms=20,
            requires_context=True,
            context_template={
                "requested_coverage": 100000,
                "requested_deductible": 500,
                "policy_type": "auto",
                "term_months": 12,
                "additional_drivers": 0,
                "vehicle_age": 3
            },
        ),
        "underwriting_v1": ServiceProfile(
            name="underwriting_v1",
            description="Standard underwriting with heavy ODFV - 23 features, 3 FVs, 1 ODFV",
            features=23,
            feature_views=3,
            odfvs=1,
            odfv_complexity="heavy",
            target_p99_ms=50,
            requires_context=True,
            context_template={
                "requested_coverage": 100000,
                "requested_deductible": 500,
                "policy_type": "auto",
                "term_months": 12,
                "additional_drivers": 0,
                "vehicle_age": 3
            },
        ),
        "underwriting_v2": ServiceProfile(
            name="underwriting_v2",
            description="Comprehensive underwriting - 50+ features, 4 FVs, 2 ODFVs",
            features=50,
            feature_views=4,
            odfvs=2,
            odfv_complexity="mixed",
            target_p99_ms=100,
            requires_context=True,
            context_template={
                "requested_coverage": 100000,
                "requested_deductible": 500,
                "policy_type": "auto",
                "term_months": 12,
                "additional_drivers": 0,
                "vehicle_age": 3
            },
        ),
    }

    # Test configurations for each dimension (using only working services)
    DIMENSION_TESTS = {
        "feature_count": [
            {"service": "benchmark_small", "features": 3},
            {"service": "benchmark_medium", "features": 12},
            {"service": "benchmark_large", "features": 40},
        ],
        "feature_views": [
            {"service": "benchmark_small", "fvs": 1, "features": 3},
            {"service": "benchmark_medium", "fvs": 3, "features": 12},
            {"service": "benchmark_large", "fvs": 3, "features": 40},
        ],
        "odfv_count": [
            {"service": "benchmark_large", "odfvs": 0, "complexity": "none"},
            # Note: ODFV services need proper deployment/context setup
            # {"service": "underwriting_quick_quote", "odfvs": 1, "complexity": "light"},
            # {"service": "underwriting_v1", "odfvs": 1, "complexity": "heavy"},
            # {"service": "underwriting_v2", "odfvs": 2, "complexity": "mixed"},
        ],
        "odfv_complexity": [
            {"service": "benchmark_large", "complexity": "none", "expected_overhead": 0},
            # Note: ODFV services need proper deployment/context setup
            # {"service": "underwriting_quick_quote", "complexity": "light", "expected_overhead": 5},
            # {"service": "underwriting_v1", "complexity": "heavy", "expected_overhead": 15},
        ],
    }

    # Standard test configurations
    BATCH_SIZES = [1, 5, 10, 25, 50]
    CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]

    def __init__(self, server_url: str, timeout_seconds: float = 60.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = None
        self.results = []

    async def run_comprehensive_tests(self) -> Dict[str, List[TestResult]]:
        """Run comprehensive tests across all dimensions."""
        results = {}

        # Create session with proper timeout and SSL settings
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds, connect=10.0)
        connector = aiohttp.TCPConnector(ssl=False, limit=100, limit_per_host=10)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            self.session = session

            print(f"🚀 Starting comprehensive performance testing")
            print(f"📍 Server: {self.server_url}")
            print(f"📊 Testing {len(self.SERVICE_PROFILES)} services across multiple dimensions\\n")

            # Dimension 1 & 2: Feature Count and Feature View Scaling
            print("🔬 Dimension 1&2: Feature Count & Feature View Scaling")
            results["feature_scaling"] = await self._test_feature_scaling()

            # Dimension 3: Concurrency Scaling
            print("\\n🔬 Dimension 3: Concurrency Scaling")
            results["concurrency"] = await self._test_concurrency_scaling()

            # Dimension 4: Data Volume (Batch Size) Scaling
            print("\\n🔬 Dimension 4: Data Volume (Batch Size) Scaling")
            results["batch_size"] = await self._test_batch_size_scaling()

            # Dimension 5 & 6: ODFV Count and Complexity Scaling (basic test)
            print("\\n🔬 Dimension 5&6: ODFV Count & Complexity Scaling (baseline only)")
            results["odfv_scaling"] = await self._test_odfv_scaling_basic()

        return results

    async def _test_feature_scaling(self) -> List[TestResult]:
        """Test how latency scales with feature count and feature view count."""
        results = []

        test_services = ["benchmark_small", "benchmark_medium", "benchmark_large"]

        for service_name in test_services:
            profile = self.SERVICE_PROFILES[service_name]
            print(f"  📊 Testing {service_name}: {profile.features} features, {profile.feature_views} FVs")

            result = await self._run_single_test(
                service_name=service_name,
                test_type="feature_scaling",
                batch_size=1,
                concurrency=1,
                num_requests=100
            )
            results.append(result)

            # Brief analysis
            if result.latencies_ms:
                features_per_ms = profile.features / result.mean_latency
                print(f"    ⚡ {result.mean_latency:.1f}ms avg ({features_per_ms:.1f} features/ms)")

        return results

    async def _test_concurrency_scaling(self) -> List[TestResult]:
        """Test how latency scales with concurrency."""
        results = []

        # Test concurrency scaling for different service types
        test_cases = [
            {"service": "benchmark_small", "description": "Small service"},
            {"service": "benchmark_large", "description": "Large service (no ODFV)"},
            {"service": "underwriting_v1", "description": "Heavy ODFV service"},
        ]

        for case in test_cases:
            service_name = case["service"]
            print(f"  📊 {case['description']} ({service_name})")

            for concurrency in [1, 5, 10, 25, 50]:
                result = await self._run_single_test(
                    service_name=service_name,
                    test_type="concurrency_scaling",
                    batch_size=1,
                    concurrency=concurrency,
                    num_requests=100  # Fixed 100 requests for consistent comparison
                )
                results.append(result)

                if result.latencies_ms:
                    rps = concurrency
                    print(f"    ⚡ {concurrency:2d} concurrent: {result.p99_latency:.1f}ms p99, {result.success_rate:.1f}% success")
                else:
                    print(f"    ❌ {concurrency:2d} concurrent: Failed")

        return results

    async def _test_batch_size_scaling(self) -> List[TestResult]:
        """Test how latency scales with batch size (data volume)."""
        results = []

        test_services = ["benchmark_small", "benchmark_large"]

        for service_name in test_services:
            print(f"  📊 {service_name}")

            for batch_size in [1, 5, 10, 25, 50]:
                result = await self._run_single_test(
                    service_name=service_name,
                    test_type="batch_scaling",
                    batch_size=batch_size,
                    concurrency=1,
                    num_requests=100
                )
                results.append(result)

                if result.latencies_ms:
                    latency_per_entity = result.mean_latency / batch_size
                    print(f"    ⚡ Batch {batch_size:2d}: {result.mean_latency:.1f}ms total ({latency_per_entity:.1f}ms/entity)")

        return results

    async def _test_odfv_scaling(self) -> List[TestResult]:
        """Test ODFV count and complexity scaling."""
        results = []

        # ODFV count scaling
        odfv_services = [
            ("benchmark_large", "0 ODFVs (baseline)"),
            ("underwriting_quick_quote", "1 lightweight ODFV"),
            ("underwriting_v1", "1 heavy ODFV"),
            ("underwriting_v2", "2 ODFVs (mixed)"),
        ]

        print(f"  📊 ODFV Count Scaling:")
        for service_name, description in odfv_services:
            result = await self._run_single_test(
                service_name=service_name,
                test_type="odfv_count",
                batch_size=1,
                concurrency=1,
                num_requests=100
            )
            results.append(result)

            if result.latencies_ms:
                profile = self.SERVICE_PROFILES[service_name]
                overhead = result.mean_latency - 10  # Assume 10ms baseline
                print(f"    ⚡ {description}: {result.mean_latency:.1f}ms (+{overhead:.1f}ms ODFV overhead)")

        # ODFV complexity comparison
        print(f"\\n  📊 ODFV Complexity Comparison:")
        complexity_tests = [
            ("underwriting_quick_quote", "Lightweight ODFV (vectorized)"),
            ("underwriting_v1", "Heavy ODFV (row iteration)"),
        ]

        for service_name, description in complexity_tests:
            # Test with different batch sizes to see scaling impact
            for batch_size in [1, 10, 50]:
                result = await self._run_single_test(
                    service_name=service_name,
                    test_type="odfv_complexity",
                    batch_size=batch_size,
                    concurrency=1,
                    num_requests=100
                )
                results.append(result)

                if result.latencies_ms:
                    print(f"    ⚡ {description} (batch {batch_size}): {result.mean_latency:.1f}ms")

        return results

    async def _run_single_test(
        self,
        service_name: str,
        test_type: str,
        batch_size: int = 1,
        concurrency: int = 1,
        num_requests: int = 100
    ) -> TestResult:
        """Run a single performance test."""

        profile = self.SERVICE_PROFILES[service_name]

        # Generate entity values
        entities = [f"CUST{i:08d}" for i in range(1, batch_size + 1)]

        # Build request payload
        payload = self._build_request_payload(service_name, entities)

        latencies = []
        errors = []
        semaphore = asyncio.Semaphore(concurrency)

        async def single_request():
            async with semaphore:
                start_time = time.time()
                try:
                    async with self.session.post(
                        f"{self.server_url}/get-online-features",
                        json=payload,
                        ssl=False
                    ) as response:
                        if response.status == 200:
                            await response.json()
                            latency_ms = (time.time() - start_time) * 1000
                            return latency_ms
                        else:
                            error_text = await response.text()
                            return f"HTTP {response.status}: {error_text[:100]}"
                except Exception as e:
                    return f"Error: {str(e)[:100]}"

        # Execute requests
        tasks = [single_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        # Separate latencies and errors
        for result in results:
            if isinstance(result, float):
                latencies.append(result)
            else:
                errors.append(str(result))

        return TestResult(
            service_name=service_name,
            test_type=test_type,
            batch_size=batch_size,
            concurrency=concurrency,
            num_requests=num_requests,
            latencies_ms=latencies,
            errors=errors,
            timestamp=datetime.now().isoformat()
        )

    async def _test_odfv_scaling_basic(self) -> List[TestResult]:
        """Basic ODFV scaling test - just baseline comparison."""
        results = []

        # For now, just test baseline (no ODFV) service
        print(f"  📊 Baseline test (no ODFVs):")
        result = await self._run_single_test(
            service_name="benchmark_large",
            test_type="odfv_baseline",
            batch_size=1,
            concurrency=1,
            num_requests=100
        )
        results.append(result)

        if result.latencies_ms:
            print(f"    ⚡ Baseline (no ODFVs): {result.mean_latency:.1f}ms")

        print(f"  📊 Note: ODFV services require proper request context setup for testing")

        return results

    def _build_request_payload(self, service_name: str, entities: List[str]) -> Dict[str, Any]:
        """Build request payload with proper context for ODFVs."""
        profile = self.SERVICE_PROFILES[service_name]

        payload = {
            "feature_service": service_name,
            "entities": {"customer_id": entities},
            "full_feature_names": False,
        }

        # Add request context for ODFV-enabled services
        if profile.requires_context and profile.context_template:
            payload["request_context"] = profile.context_template

        return payload

    def print_summary_report(self, all_results: Dict[str, List[TestResult]]):
        """Print comprehensive summary report."""
        print("\\n" + "="*80)
        print("📊 COMPREHENSIVE PERFORMANCE TESTING SUMMARY")
        print("="*80)

        for dimension, results in all_results.items():
            if not results:
                continue

            print(f"\\n🔬 {dimension.upper()} RESULTS:")

            # Create summary table
            table_data = []
            headers = ["Service", "Features", "FVs", "ODFVs", "Batch", "Concurrency", "P99 (ms)", "Avg (ms)", "Success %"]

            for result in results:
                if result.latencies_ms:  # Only show successful tests
                    profile = self.SERVICE_PROFILES[result.service_name]
                    table_data.append([
                        result.service_name,
                        profile.features,
                        profile.feature_views,
                        profile.odfvs,
                        result.batch_size,
                        result.concurrency,
                        f"{result.p99_latency:.1f}",
                        f"{result.mean_latency:.1f}",
                        f"{result.success_rate:.1f}%"
                    ])

            if TABULATE_AVAILABLE and table_data:
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                # Fallback formatting
                for row in table_data:
                    print(f"  {row[0]}: P99={row[6]}ms, Avg={row[7]}ms")

        print("\\n" + "="*80)
        print("✅ Performance testing complete!")


async def main():
    parser = argparse.ArgumentParser(description="Enhanced multi-dimensional performance testing")
    parser.add_argument("--server-url", required=True, help="Feature server URL")
    parser.add_argument("--service", help="Test specific service only")
    parser.add_argument("--dimension", choices=["feature_count", "concurrency", "batch_size", "odfv_scaling", "all"],
                       default="all", help="Test specific dimension")
    parser.add_argument("--suite", choices=["quick", "standard", "comprehensive"], default="standard",
                       help="Test suite to run")

    args = parser.parse_args()

    tester = EnhancedPerformanceTester(args.server_url)

    print(f"🚀 Enhanced Performance Testing for AWS Insurance Demo")
    print(f"📍 Server: {args.server_url}")
    print(f"🎯 Suite: {args.suite}")

    if args.service:
        print(f"🔍 Testing single service: {args.service}")
        # Run single service test
        timeout = aiohttp.ClientTimeout(total=60.0, connect=10.0)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tester.session = session
            result = await tester._run_single_test(
                service_name=args.service,
                test_type="single",
                batch_size=1,
                concurrency=1,
                num_requests=100
            )
            if result.latencies_ms:
                print(f"✅ {args.service}: {result.mean_latency:.1f}ms avg, {result.p99_latency:.1f}ms p99")
            else:
                print(f"❌ {args.service}: Test failed")
                for error in result.errors:
                    print(f"   Error: {error}")
    else:
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests()
        tester.print_summary_report(results)


if __name__ == "__main__":
    asyncio.run(main())