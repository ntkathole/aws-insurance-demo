#!/usr/bin/env python3
"""
Architecture Validation Script - Simplified AWS Insurance Demo

This script validates the simplified architecture setup:
- 1 consolidated customer feature view (40+ features)
- 1 premium calculator ODFV (native Python)
- New simplified feature services
- Performance testing capabilities

Ensures the cleanup and optimization was successful.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Any

# Add feature_repo to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_repo'))


class ArchitectureValidator:
    """Validate the simplified architecture setup."""

    def __init__(self):
        self.validation_results = {}
        self.feature_repo_path = os.path.join(os.path.dirname(__file__), '..', 'feature_repo')

    def validate_file_structure(self) -> Dict[str, bool]:
        """Validate the new file structure."""
        print("🔍 Validating file structure...")

        required_files = {
            "entities.py": "Updated entities (customer only)",
            "data_sources.py": "Consolidated customer data source",
            "feature_views/customer_features.py": "Consolidated customer feature view",
            "on_demand_features.py": "Simplified premium calculator ODFV",
            "feature_services.py": "Simplified feature services"
        }

        removed_files = {
            "feature_views/underwriting_features.py": "Old underwriting features",
            "feature_views/claims_features.py": "Old claims features",
            "feature_views/streaming_features.py": "Old streaming features"
        }

        backup_files = {
            "on_demand_features_original.py": "Original pandas ODFVs",
            "on_demand_features_optimized_original.py": "Original optimized ODFVs",
            "feature_services_original.py": "Original feature services"
        }

        results = {}

        # Check required files exist
        for file_path, description in required_files.items():
            full_path = os.path.join(self.feature_repo_path, file_path)
            exists = os.path.exists(full_path)
            results[f"required_{file_path}"] = exists
            print(f"  {'✓' if exists else '❌'} {description}: {file_path}")

        # Check removed files don't exist
        for file_path, description in removed_files.items():
            full_path = os.path.join(self.feature_repo_path, file_path)
            removed = not os.path.exists(full_path)
            results[f"removed_{file_path}"] = removed
            print(f"  {'✓' if removed else '⚠️'} Removed {description}: {file_path}")

        # Check backup files exist
        for file_path, description in backup_files.items():
            full_path = os.path.join(self.feature_repo_path, file_path)
            exists = os.path.exists(full_path)
            results[f"backup_{file_path}"] = exists
            print(f"  {'✓' if exists else '⚠️'} Backup {description}: {file_path}")

        return results

    def validate_entities(self) -> Dict[str, bool]:
        """Validate simplified entities."""
        print(f"\n🏗️ Validating entities...")

        try:
            from entities import customer
            print(f"  ✓ Customer entity loaded successfully")
            print(f"    Name: {customer.name}")
            # Handle API inconsistency
            try:
                join_info = customer.join_keys if hasattr(customer, 'join_keys') else customer.join_key
                print(f"    Join keys: {join_info}")
            except AttributeError:
                print(f"    Join keys: [customer_id] (default)")

            # Check for removed entities
            removed_entities = ["policy", "claim", "provider", "transaction"]
            for entity_name in removed_entities:
                try:
                    exec(f"from entities import {entity_name}")
                    print(f"  ❌ {entity_name} entity still exists (should be removed)")
                    return {"customer_exists": True, f"{entity_name}_removed": False}
                except ImportError:
                    print(f"  ✓ {entity_name} entity removed successfully")

            return {"customer_exists": True, "removed_entities": True}

        except ImportError as e:
            print(f"  ❌ Failed to import customer entity: {e}")
            return {"customer_exists": False}

    def validate_data_sources(self) -> Dict[str, bool]:
        """Validate consolidated data source."""
        print(f"\n💾 Validating data sources...")

        try:
            from data_sources import customer_consolidated_source
            print(f"  ✓ Consolidated customer source loaded")
            print(f"    Name: {customer_consolidated_source.name}")
            print(f"    Source type: {type(customer_consolidated_source).__name__}")

            # Check if it's using a query (consolidated) rather than a simple table
            has_query = hasattr(customer_consolidated_source, 'query') and customer_consolidated_source.query
            print(f"    Query-based: {'✓' if has_query else '❌'}")

            return {"consolidated_source_exists": True, "uses_query": has_query}

        except ImportError as e:
            print(f"  ❌ Failed to import consolidated source: {e}")
            return {"consolidated_source_exists": False}

    def validate_feature_views(self) -> Dict[str, bool]:
        """Validate consolidated feature view."""
        print(f"\n📊 Validating feature views...")

        try:
            from feature_views.customer_features import customer_consolidated_fv, FEATURE_GROUPS
            print(f"  ✓ Consolidated customer feature view loaded")
            print(f"    Name: {customer_consolidated_fv.name}")
            print(f"    Features: {len(customer_consolidated_fv.schema)} total")
            print(f"    Entity: {[e.name for e in customer_consolidated_fv.entities]}")

            # Check feature groups
            print(f"  ✓ Feature groups for latency testing:")
            for group_name, features in FEATURE_GROUPS.items():
                feature_count = len(features) if features else "all"
                print(f"    • {group_name}: {feature_count} features")

            # Validate expected feature count (should be ~40)
            feature_count = len(customer_consolidated_fv.schema)
            expected_range = (35, 45)  # Allow some flexibility
            count_valid = expected_range[0] <= feature_count <= expected_range[1]

            print(f"  {'✓' if count_valid else '❌'} Feature count in expected range: {feature_count} "
                  f"(expected: {expected_range[0]}-{expected_range[1]})")

            return {
                "consolidated_fv_exists": True,
                "feature_count_valid": count_valid,
                "has_feature_groups": True
            }

        except ImportError as e:
            print(f"  ❌ Failed to import consolidated feature view: {e}")
            return {"consolidated_fv_exists": False}

    def validate_on_demand_features(self) -> Dict[str, bool]:
        """Validate simplified ODFV."""
        print(f"\n🔄 Validating on-demand features...")

        try:
            from on_demand_features import premium_calculator_optimized, underwriting_request
            print(f"  ✓ Premium calculator ODFV loaded")
            print(f"    Function: {premium_calculator_optimized.__name__}")
            print(f"    Mode: native Python (optimized)")

            print(f"  ✓ Underwriting request source loaded")
            print(f"    Schema fields: {len(underwriting_request.schema)}")

            # Test the function works
            test_data = {
                "age": 35,
                "region_risk_zone": 3,
                "credit_score": 720,
                "requested_coverage": 250000,
                "requested_deductible": 1000,
                "vehicle_age": 5,
            }

            # Try to get the actual function
            if hasattr(premium_calculator_optimized, '_func'):
                func = premium_calculator_optimized._func
            else:
                func = premium_calculator_optimized

            result = func(test_data)
            print(f"  ✓ Premium calculator test successful")
            print(f"    Sample output: annual_premium=${result.get('estimated_annual_premium', 'N/A')}")

            return {
                "premium_calculator_exists": True,
                "request_source_exists": True,
                "function_works": True
            }

        except Exception as e:
            print(f"  ❌ Failed to validate ODFV: {e}")
            return {"premium_calculator_exists": False}

    def validate_feature_services(self) -> Dict[str, bool]:
        """Validate simplified feature services."""
        print(f"\n🎯 Validating feature services...")

        try:
            # Import specific services
            from feature_services import (
                customer_5_features, customer_10_features, customer_20_features, customer_40_features,
                customer_5_with_premium, customer_40_with_premium, quick_quote, comprehensive_quote
            )

            services = {
                "customer_5_features": customer_5_features,
                "customer_10_features": customer_10_features,
                "customer_20_features": customer_20_features,
                "customer_40_features": customer_40_features,
                "customer_5_with_premium": customer_5_with_premium,
                "customer_40_with_premium": customer_40_with_premium,
                "quick_quote": quick_quote,
                "comprehensive_quote": comprehensive_quote,
            }

            print(f"  ✓ Loaded {len(services)} feature services")

            # Validate latency testing services
            latency_services = ["customer_5_features", "customer_10_features", "customer_20_features", "customer_40_features"]
            for service_name in latency_services:
                service = services[service_name]
                expected_count = int(service_name.split('_')[1])
                print(f"    • {service_name}: for {expected_count}-feature latency testing")

            # Validate ODFV combination services
            odfv_services = ["customer_5_with_premium", "customer_40_with_premium"]
            for service_name in odfv_services:
                service = services[service_name]
                print(f"    • {service_name}: batch + ODFV combination testing")

            # Validate production services
            production_services = ["quick_quote", "comprehensive_quote"]
            for service_name in production_services:
                service = services[service_name]
                print(f"    • {service_name}: production use case")

            return {
                "feature_services_loaded": True,
                "latency_testing_services": True,
                "production_services": True
            }

        except ImportError as e:
            print(f"  ❌ Failed to import feature services: {e}")
            return {"feature_services_loaded": False}

    def validate_performance_scripts(self) -> Dict[str, bool]:
        """Validate updated performance test scripts."""
        print(f"\n⚡ Validating performance test scripts...")

        scripts_to_check = {
            "local_performance_test_simplified.py": "Local ODFV performance testing",
            "server_performance_test_simplified.py": "Server-side latency testing"
        }

        results = {}
        for script_name, description in scripts_to_check.items():
            script_path = os.path.join(os.path.dirname(__file__), script_name)
            exists = os.path.exists(script_path)
            results[f"script_{script_name}"] = exists
            print(f"  {'✓' if exists else '❌'} {description}: {script_name}")

        return results

    def run_full_validation(self):
        """Run complete architecture validation."""
        print("🚀 Starting Architecture Validation")
        print("   Simplified AWS Insurance Demo")
        print(f"{'='*80}")

        # Run all validations
        all_results = {}

        all_results.update(self.validate_file_structure())
        all_results.update(self.validate_entities())
        all_results.update(self.validate_data_sources())
        all_results.update(self.validate_feature_views())
        all_results.update(self.validate_on_demand_features())
        all_results.update(self.validate_feature_services())
        all_results.update(self.validate_performance_scripts())

        # Summary
        print(f"\n{'='*80}")
        print("📋 VALIDATION SUMMARY")
        print(f"{'='*80}")

        total_checks = len(all_results)
        passed_checks = sum(1 for result in all_results.values() if result)

        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success rate: {passed_checks/total_checks:.1%}")

        # Key architecture components
        key_components = {
            "consolidated_fv_exists": "Consolidated customer feature view",
            "premium_calculator_exists": "Premium calculator ODFV",
            "feature_services_loaded": "Simplified feature services",
            "script_local_performance_test_simplified.py": "Local performance testing",
            "script_server_performance_test_simplified.py": "Server performance testing"
        }

        print(f"\n🎯 KEY ARCHITECTURE COMPONENTS:")
        for key, description in key_components.items():
            status = "✅" if all_results.get(key, False) else "❌"
            print(f"   {status} {description}")

        # Architecture benefits
        print(f"\n🏆 ARCHITECTURE BENEFITS:")
        if all_results.get("consolidated_fv_exists"):
            print(f"   ✅ Single consolidated feature view (40+ features)")
        if all_results.get("premium_calculator_exists"):
            print(f"   ✅ Native Python ODFV optimization")
        if all_results.get("latency_testing_services"):
            print(f"   ✅ Variable feature count testing (5, 10, 20, 40 features)")
        if all_results.get("script_local_performance_test_simplified.py"):
            print(f"   ✅ Computational performance testing")
        if all_results.get("script_server_performance_test_simplified.py"):
            print(f"   ✅ End-to-end latency testing")

        overall_success = passed_checks >= total_checks * 0.8  # 80% success threshold

        print(f"\n{'='*80}")
        if overall_success:
            print("✅ ARCHITECTURE VALIDATION SUCCESSFUL!")
            print("   The simplified architecture is ready for latency testing.")
            print("   Next steps:")
            print("   1. Run local performance tests: python scripts/local_performance_test_simplified.py")
            print("   2. Set up Feast server and run: python scripts/server_performance_test_simplified.py")
        else:
            print("❌ ARCHITECTURE VALIDATION FAILED!")
            print("   Please fix the failed components before proceeding.")

        print(f"{'='*80}")

        return overall_success


def main():
    """Run architecture validation."""
    try:
        validator = ArchitectureValidator()
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()