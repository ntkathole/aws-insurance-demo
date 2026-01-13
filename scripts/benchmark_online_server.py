#!/usr/bin/env python
"""
Online Server Latency Benchmark for AWS Insurance Demo.

This script benchmarks the Feast feature server for insurance use cases,
measuring latency across different scenarios:
- Underwriting (Real-Time PCM): Target < 50ms p99
- Claims (Batch): Target < 200ms p99
- Fraud Detection (DSS): Target < 30ms p99

Usage:
    # Quick benchmark with default settings
    python benchmark_online_server.py --server-url http://localhost:6566
    
    # Full benchmark suite
    python benchmark_online_server.py --server-url http://localhost:6566 --suite full
    
    # Custom benchmark
    python benchmark_online_server.py --server-url http://localhost:6566 \
        --feature-service underwriting_v2 \
        --entity-key customer_id \
        --batch-sizes 1,10,50,100

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


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    feature_service: str
    num_features: int
    batch_size: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
    
    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[len(sorted_l) // 2]
    
    @property
    def p90_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[int(len(sorted_l) * 0.9)]
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[int(len(sorted_l) * 0.95)]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = min(int(len(sorted_l) * 0.99), len(sorted_l) - 1)
        return sorted_l[idx]
    
    @property
    def throughput_rps(self) -> float:
        return (1000.0 / self.mean_latency) if self.mean_latency > 0 else 0.0


class InsuranceBenchmark:
    """Benchmark runner for insurance feature store."""
    
    # Pre-defined benchmark scenarios for insurance use cases
    SCENARIOS = {
        "underwriting_quick": {
            "description": "Quick quote - minimal features",
            "feature_service": "underwriting_quick_quote",
            "entity_key": "customer_id",
            "target_p99_ms": 20,
            "use_case": "PCM",
        },
        "underwriting_v1": {
            "description": "Basic underwriting",
            "feature_service": "underwriting_v1",
            "entity_key": "customer_id",
            "target_p99_ms": 50,
            "use_case": "PCM",
        },
        "underwriting_v2": {
            "description": "Comprehensive underwriting with ODFV",
            "feature_service": "underwriting_v2",
            "entity_key": "customer_id",
            "target_p99_ms": 100,
            "use_case": "PCM",
        },
        "claims_v1": {
            "description": "Claims assessment",
            "feature_service": "claims_assessment_v1",
            "entity_key": "customer_id",  # Note: This uses claim_id in practice
            "target_p99_ms": 200,
            "use_case": "Batch",
        },
        "fraud_v1": {
            "description": "Transaction fraud detection",
            "feature_service": "fraud_detection_v1",
            "entity_key": "transaction_id",
            "target_p99_ms": 30,
            "use_case": "DSS",
        },
        "benchmark_small": {
            "description": "Small feature set benchmark",
            "feature_service": "benchmark_small",
            "entity_key": "customer_id",
            "target_p99_ms": 10,
            "use_case": "Benchmark",
        },
        "benchmark_medium": {
            "description": "Medium feature set benchmark",
            "feature_service": "benchmark_medium",
            "entity_key": "customer_id",
            "target_p99_ms": 30,
            "use_case": "Benchmark",
        },
        "benchmark_large": {
            "description": "Large feature set benchmark",
            "feature_service": "benchmark_large",
            "entity_key": "customer_id",
            "target_p99_ms": 75,
            "use_case": "Benchmark",
        },
    }
    
    BENCHMARK_SUITES = {
        "quick": ["benchmark_small", "underwriting_quick"],
        "standard": ["benchmark_small", "benchmark_medium", "underwriting_v1"],
        "full": [
            "benchmark_small", "benchmark_medium", "benchmark_large",
            "underwriting_quick", "underwriting_v1", "underwriting_v2",
        ],
        "pcm": ["underwriting_quick", "underwriting_v1", "underwriting_v2"],
        "claims": ["claims_v1"],
        "fraud": ["fraud_v1"],
    }
    
    def __init__(
        self,
        server_url: str,
        num_requests: int = 100,
        concurrency: int = 10,
        warmup_requests: int = 10,
        timeout_seconds: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.warmup_requests = warmup_requests
        self.timeout_seconds = timeout_seconds
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    def _generate_entity_values(
        self,
        entity_key: str,
        batch_size: int,
        num_entities: int = 10000
    ) -> List[Any]:
        """Generate entity values for benchmarking."""
        if entity_key == "customer_id":
            return [f"CUST{random.randint(1, num_entities):08d}" for _ in range(batch_size)]
        elif entity_key == "claim_id":
            return [f"CLM{random.randint(1, num_entities):010d}" for _ in range(batch_size)]
        elif entity_key == "transaction_id":
            return [f"TXN{random.randint(1, num_entities):012d}" for _ in range(batch_size)]
        elif entity_key == "policy_id":
            return [f"POL{random.randint(1, num_entities):08d}" for _ in range(batch_size)]
        else:
            return list(range(1, batch_size + 1))
    
    async def _make_request(
        self,
        feature_service: str,
        entity_key: str,
        entity_values: List[Any],
    ) -> tuple[float, bool, Optional[str]]:
        """Make a single request to the feature server."""
        url = f"{self.server_url}/get-online-features"
        payload = {
            "feature_service": feature_service,
            "entities": {entity_key: entity_values},
            "full_feature_names": False,
        }
        
        start_time = time.perf_counter()
        
        try:
            async with self._session.post(url, json=payload) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    await response.json()
                    return latency_ms, True, None
                else:
                    error_text = await response.text()
                    return latency_ms, False, f"HTTP {response.status}: {error_text[:100]}"
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return latency_ms, False, "Request timeout"
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return latency_ms, False, str(e)[:100]
    
    async def _run_concurrent_requests(
        self,
        feature_service: str,
        entity_key: str,
        batch_size: int,
        num_requests: int,
    ) -> List[tuple[float, bool, Optional[str]]]:
        """Run multiple requests with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def limited_request():
            async with semaphore:
                entity_values = self._generate_entity_values(entity_key, batch_size)
                return await self._make_request(feature_service, entity_key, entity_values)
        
        tasks = [limited_request() for _ in range(num_requests)]
        return await asyncio.gather(*tasks)
    
    async def run_scenario(
        self,
        scenario_name: str,
        batch_sizes: List[int],
    ) -> List[BenchmarkResult]:
        """Run a benchmark scenario with multiple batch sizes."""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.SCENARIOS[scenario_name]
        results = []
        
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"Description: {scenario['description']}")
        print(f"Use Case: {scenario['use_case']}")
        print(f"Target P99: {scenario['target_p99_ms']}ms")
        print(f"{'='*70}")
        
        for batch_size in batch_sizes:
            name = f"{scenario_name}_batch{batch_size}"
            
            print(f"\n  Testing batch_size={batch_size}...")
            
            # Warmup
            print(f"    Warmup ({self.warmup_requests} requests)...")
            await self._run_concurrent_requests(
                scenario["feature_service"],
                scenario["entity_key"],
                batch_size,
                self.warmup_requests,
            )
            
            # Benchmark
            print(f"    Benchmark ({self.num_requests} requests)...")
            raw_results = await self._run_concurrent_requests(
                scenario["feature_service"],
                scenario["entity_key"],
                batch_size,
                self.num_requests,
            )
            
            # Process results
            latencies = []
            errors = []
            successful = 0
            failed = 0
            
            for latency, success, error in raw_results:
                if success:
                    latencies.append(latency)
                    successful += 1
                else:
                    failed += 1
                    if error:
                        errors.append(error)
            
            result = BenchmarkResult(
                name=name,
                feature_service=scenario["feature_service"],
                num_features=0,  # Would need to query registry for actual count
                batch_size=batch_size,
                total_requests=len(raw_results),
                successful_requests=successful,
                failed_requests=failed,
                latencies_ms=latencies,
                errors=errors[:5],
            )
            results.append(result)
            
            # Print quick summary
            status = "✓" if result.p99_latency <= scenario["target_p99_ms"] else "✗"
            print(f"    {status} p99={result.p99_latency:.2f}ms (target: {scenario['target_p99_ms']}ms)")
        
        return results
    
    async def run_suite(
        self,
        suite_name: str,
        batch_sizes: List[int],
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run a predefined benchmark suite."""
        if suite_name not in self.BENCHMARK_SUITES:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(self.BENCHMARK_SUITES.keys())}")
        
        scenarios = self.BENCHMARK_SUITES[suite_name]
        all_results = {}
        
        for scenario_name in scenarios:
            results = await self.run_scenario(scenario_name, batch_sizes)
            all_results[scenario_name] = results
        
        return all_results
    
    async def run_custom(
        self,
        feature_service: str,
        entity_key: str,
        batch_sizes: List[int],
    ) -> List[BenchmarkResult]:
        """Run a custom benchmark with specified parameters."""
        results = []
        
        print(f"\n{'='*70}")
        print(f"Custom Benchmark")
        print(f"Feature Service: {feature_service}")
        print(f"Entity Key: {entity_key}")
        print(f"{'='*70}")
        
        for batch_size in batch_sizes:
            name = f"custom_batch{batch_size}"
            
            print(f"\n  Testing batch_size={batch_size}...")
            
            # Warmup
            await self._run_concurrent_requests(
                feature_service, entity_key, batch_size, self.warmup_requests
            )
            
            # Benchmark
            raw_results = await self._run_concurrent_requests(
                feature_service, entity_key, batch_size, self.num_requests
            )
            
            latencies = [r[0] for r in raw_results if r[1]]
            errors = [r[2] for r in raw_results if not r[1] and r[2]]
            
            result = BenchmarkResult(
                name=name,
                feature_service=feature_service,
                num_features=0,
                batch_size=batch_size,
                total_requests=len(raw_results),
                successful_requests=len(latencies),
                failed_requests=len(errors),
                latencies_ms=latencies,
                errors=errors[:5],
            )
            results.append(result)
            
            print(f"    p50={result.p50_latency:.2f}ms, p99={result.p99_latency:.2f}ms")
        
        return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    headers = [
        "Name", "Service", "Batch", "Success%",
        "Mean(ms)", "P50(ms)", "P95(ms)", "P99(ms)", "RPS"
    ]
    
    rows = []
    for r in results:
        rows.append([
            r.name[:30],
            r.feature_service[:25],
            r.batch_size,
            f"{r.success_rate:.1f}",
            f"{r.mean_latency:.2f}",
            f"{r.p50_latency:.2f}",
            f"{r.p95_latency:.2f}",
            f"{r.p99_latency:.2f}",
            f"{r.throughput_rps:.1f}",
        ])
    
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    
    if TABULATE_AVAILABLE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(" | ".join(headers))
        print("-" * 100)
        for row in rows:
            print(" | ".join(str(x) for x in row))


def print_latency_summary(all_results: Dict[str, List[BenchmarkResult]]):
    """Print a summary of latency vs batch size."""
    print("\n" + "=" * 100)
    print("LATENCY SUMMARY (P99 in ms)")
    print("=" * 100)
    
    # Collect all batch sizes
    batch_sizes = set()
    for results in all_results.values():
        for r in results:
            batch_sizes.add(r.batch_size)
    batch_sizes = sorted(batch_sizes)
    
    if TABULATE_AVAILABLE:
        headers = ["Scenario"] + [f"Batch={b}" for b in batch_sizes]
        rows = []
        
        for scenario_name, results in all_results.items():
            row = [scenario_name[:25]]
            by_batch = {r.batch_size: r.p99_latency for r in results}
            for bs in batch_sizes:
                val = by_batch.get(bs)
                row.append(f"{val:.2f}" if val else "-")
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name}:")
            for r in results:
                print(f"  Batch {r.batch_size}: p99={r.p99_latency:.2f}ms")


def save_results_json(all_results: Dict[str, List[BenchmarkResult]], output_file: str):
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {}
    }
    
    for scenario_name, results in all_results.items():
        data["scenarios"][scenario_name] = [
            {
                "name": r.name,
                "feature_service": r.feature_service,
                "batch_size": r.batch_size,
                "total_requests": r.total_requests,
                "successful_requests": r.successful_requests,
                "failed_requests": r.failed_requests,
                "success_rate": r.success_rate,
                "mean_latency_ms": r.mean_latency,
                "p50_latency_ms": r.p50_latency,
                "p95_latency_ms": r.p95_latency,
                "p99_latency_ms": r.p99_latency,
                "throughput_rps": r.throughput_rps,
            }
            for r in results
        ]
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Feast feature server for insurance use cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark
  python benchmark_online_server.py --server-url http://localhost:6566
  
  # Full benchmark suite
  python benchmark_online_server.py --server-url http://localhost:6566 --suite full
  
  # Custom benchmark
  python benchmark_online_server.py --server-url http://localhost:6566 \\
      --feature-service underwriting_v2 \\
      --entity-key customer_id \\
      --batch-sizes 1,10,50,100

Available Suites: quick, standard, full, pcm, claims, fraud
Available Scenarios: underwriting_quick, underwriting_v1, underwriting_v2,
                    claims_v1, fraud_v1, benchmark_small, benchmark_medium, benchmark_large
        """
    )
    
    parser.add_argument(
        "--server-url",
        required=True,
        help="Feature server URL (e.g., http://localhost:6566)"
    )
    parser.add_argument(
        "--suite",
        choices=list(InsuranceBenchmark.BENCHMARK_SUITES.keys()),
        help="Pre-defined benchmark suite to run"
    )
    parser.add_argument(
        "--scenario",
        choices=list(InsuranceBenchmark.SCENARIOS.keys()),
        help="Single scenario to run"
    )
    parser.add_argument(
        "--feature-service",
        help="Custom feature service to benchmark"
    )
    parser.add_argument(
        "--entity-key",
        default="customer_id",
        help="Entity key name (default: customer_id)"
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,10,50,100",
        help="Comma-separated batch sizes (default: 1,10,50,100)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests per configuration (default: 100)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for JSON results"
    )
    
    args = parser.parse_args()
    
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    print("\n" + "=" * 70)
    print("AWS INSURANCE DEMO - FEATURE SERVER BENCHMARK")
    print("=" * 70)
    print(f"Server URL: {args.server_url}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Requests per config: {args.num_requests}")
    print(f"Concurrency: {args.concurrency}")
    
    async with InsuranceBenchmark(
        server_url=args.server_url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        warmup_requests=args.warmup,
    ) as benchmark:
        
        all_results = {}
        
        if args.suite:
            all_results = await benchmark.run_suite(args.suite, batch_sizes)
        elif args.scenario:
            results = await benchmark.run_scenario(args.scenario, batch_sizes)
            all_results[args.scenario] = results
        elif args.feature_service:
            results = await benchmark.run_custom(
                args.feature_service,
                args.entity_key,
                batch_sizes
            )
            all_results["custom"] = results
        else:
            # Default: run quick suite
            all_results = await benchmark.run_suite("quick", batch_sizes)
        
        # Print all results
        all_flat_results = []
        for results in all_results.values():
            all_flat_results.extend(results)
        
        print_results_table(all_flat_results)
        print_latency_summary(all_results)
        
        if args.output:
            save_results_json(all_results, args.output)
        
        # Print any errors
        all_errors = []
        for results in all_results.values():
            for r in results:
                all_errors.extend(r.errors)
        
        if all_errors:
            print("\n" + "=" * 70)
            print("ERRORS (sample)")
            print("=" * 70)
            for error in all_errors[:5]:
                print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
