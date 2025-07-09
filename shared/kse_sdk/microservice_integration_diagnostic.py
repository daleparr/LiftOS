#!/usr/bin/env python3
"""
KSE-SDK Microservice Integration Diagnostic Tool
Tests bidirectional KSE access for all LiftOS microservices
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class MicroserviceKSEDiagnostic:
    """Diagnostic tool for microservice KSE integration"""
    
    def __init__(self):
        self.project_root = Path(project_root)
        self.services_dir = self.project_root / "services"
        self.results = {}
        
    def get_microservices(self) -> List[str]:
        """Get list of all microservices"""
        services = []
        if self.services_dir.exists():
            for item in self.services_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    services.append(item.name)
        return sorted(services)
    
    def check_kse_imports(self, service_name: str) -> Dict[str, Any]:
        """Check if service can import KSE components"""
        result = {
            "can_import_kse": False,
            "kse_imports": [],
            "import_errors": [],
            "files_checked": []
        }
        
        service_dir = self.services_dir / service_name
        if not service_dir.exists():
            result["import_errors"].append(f"Service directory not found: {service_dir}")
            return result
        
        # Look for Python files that might import KSE
        python_files = list(service_dir.rglob("*.py"))
        result["files_checked"] = [str(f.relative_to(service_dir)) for f in python_files]
        
        kse_import_patterns = [
            "from shared.kse_sdk",
            "import shared.kse_sdk",
            "from ..shared.kse_sdk",
            "KSEMemory",
            "LiftKSEClient",
            "kse_client"
        ]
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in kse_import_patterns:
                    if pattern in content:
                        result["kse_imports"].append({
                            "file": str(py_file.relative_to(service_dir)),
                            "pattern": pattern,
                            "line_count": content.count(pattern)
                        })
                        result["can_import_kse"] = True
            except Exception as e:
                result["import_errors"].append(f"Error reading {py_file}: {str(e)}")
        
        return result
    
    def check_kse_functionality(self, service_name: str) -> Dict[str, Any]:
        """Check if service has KSE read/write functionality"""
        result = {
            "has_read_operations": False,
            "has_write_operations": False,
            "has_trace_operations": False,
            "operations_found": [],
            "files_with_operations": []
        }
        
        service_dir = self.services_dir / service_name
        if not service_dir.exists():
            return result
        
        # Patterns indicating KSE operations
        read_patterns = [
            "search", "query", "retrieve", "get", "find", "fetch",
            ".search(", ".query(", ".get(", ".retrieve("
        ]
        
        write_patterns = [
            "store", "save", "insert", "upsert", "update", "create",
            ".store(", ".save(", ".insert(", ".upsert(", ".update("
        ]
        
        trace_patterns = [
            "trace", "log", "observe", "monitor", "track",
            "analytics", "metrics", "feedback"
        ]
        
        python_files = list(service_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8').lower()
                file_ops = []
                
                # Check for read operations
                for pattern in read_patterns:
                    if pattern in content:
                        result["has_read_operations"] = True
                        file_ops.append(f"READ: {pattern}")
                
                # Check for write operations
                for pattern in write_patterns:
                    if pattern in content:
                        result["has_write_operations"] = True
                        file_ops.append(f"WRITE: {pattern}")
                
                # Check for trace operations
                for pattern in trace_patterns:
                    if pattern in content:
                        result["has_trace_operations"] = True
                        file_ops.append(f"TRACE: {pattern}")
                
                if file_ops:
                    result["files_with_operations"].append({
                        "file": str(py_file.relative_to(service_dir)),
                        "operations": file_ops
                    })
                    result["operations_found"].extend(file_ops)
                    
            except Exception as e:
                continue
        
        return result
    
    def check_service_main_files(self, service_name: str) -> Dict[str, Any]:
        """Check main service files for KSE integration"""
        result = {
            "has_main_file": False,
            "main_files": [],
            "kse_integration_score": 0,
            "integration_details": []
        }
        
        service_dir = self.services_dir / service_name
        if not service_dir.exists():
            return result
        
        # Look for main files
        main_file_patterns = ["app.py", "main.py", "server.py", "__init__.py"]
        
        for pattern in main_file_patterns:
            main_file = service_dir / pattern
            if main_file.exists():
                result["has_main_file"] = True
                result["main_files"].append(pattern)
                
                try:
                    content = main_file.read_text(encoding='utf-8')
                    
                    # Score KSE integration
                    score = 0
                    details = []
                    
                    if "kse" in content.lower():
                        score += 2
                        details.append("Contains KSE references")
                    
                    if "shared.kse_sdk" in content:
                        score += 3
                        details.append("Imports KSE SDK")
                    
                    if "KSEMemory" in content or "LiftKSEClient" in content:
                        score += 3
                        details.append("Uses KSE client/memory")
                    
                    if "search" in content.lower() or "query" in content.lower():
                        score += 1
                        details.append("Has search/query operations")
                    
                    if "store" in content.lower() or "save" in content.lower():
                        score += 1
                        details.append("Has store/save operations")
                    
                    result["kse_integration_score"] += score
                    result["integration_details"].extend(details)
                    
                except Exception as e:
                    result["integration_details"].append(f"Error reading {pattern}: {str(e)}")
        
        return result
    
    def diagnose_service(self, service_name: str) -> Dict[str, Any]:
        """Complete diagnostic for a single service"""
        print(f"\n[*] Diagnosing {service_name}...")
        
        result = {
            "service_name": service_name,
            "imports": self.check_kse_imports(service_name),
            "functionality": self.check_kse_functionality(service_name),
            "main_files": self.check_service_main_files(service_name),
            "overall_score": 0,
            "recommendations": []
        }
        
        # Calculate overall score
        score = 0
        
        if result["imports"]["can_import_kse"]:
            score += 3
        
        if result["functionality"]["has_read_operations"]:
            score += 2
        
        if result["functionality"]["has_write_operations"]:
            score += 2
        
        if result["functionality"]["has_trace_operations"]:
            score += 1
        
        score += result["main_files"]["kse_integration_score"]
        
        result["overall_score"] = score
        
        # Generate recommendations
        recommendations = []
        
        if not result["imports"]["can_import_kse"]:
            recommendations.append("Add KSE SDK imports to enable universal substrate access")
        
        if not result["functionality"]["has_read_operations"]:
            recommendations.append("Implement KSE read operations to retrieve intelligence data")
        
        if not result["functionality"]["has_write_operations"]:
            recommendations.append("Implement KSE write operations to enrich intelligence layer")
        
        if not result["functionality"]["has_trace_operations"]:
            recommendations.append("Add tracing/observability to track service interactions")
        
        if result["main_files"]["kse_integration_score"] < 5:
            recommendations.append("Enhance main service files with comprehensive KSE integration")
        
        result["recommendations"] = recommendations
        
        # Print summary
        status = "[+] GOOD" if score >= 8 else "[~] PARTIAL" if score >= 4 else "[-] POOR"
        print(f"    {status} Integration Score: {score}/10")
        
        if result["imports"]["can_import_kse"]:
            print(f"    [+] KSE Imports: {len(result['imports']['kse_imports'])} found")
        else:
            print(f"    [-] KSE Imports: None found")
        
        if result["functionality"]["has_read_operations"]:
            print(f"    [+] Read Operations: Available")
        else:
            print(f"    [-] Read Operations: Missing")
        
        if result["functionality"]["has_write_operations"]:
            print(f"    [+] Write Operations: Available")
        else:
            print(f"    [-] Write Operations: Missing")
        
        return result
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run diagnostic on all microservices"""
        print("*** KSE-SDK Microservice Integration Diagnostic Tool")
        print("=" * 70)
        
        services = self.get_microservices()
        print(f"Found {len(services)} microservices: {', '.join(services)}")
        
        results = {}
        total_score = 0
        max_score = 0
        
        for service in services:
            service_result = self.diagnose_service(service)
            results[service] = service_result
            total_score += service_result["overall_score"]
            max_score += 10
        
        # Generate summary
        summary = {
            "total_services": len(services),
            "services_with_kse": sum(1 for r in results.values() if r["imports"]["can_import_kse"]),
            "services_with_read": sum(1 for r in results.values() if r["functionality"]["has_read_operations"]),
            "services_with_write": sum(1 for r in results.values() if r["functionality"]["has_write_operations"]),
            "services_with_trace": sum(1 for r in results.values() if r["functionality"]["has_trace_operations"]),
            "average_score": total_score / len(services) if services else 0,
            "total_score": total_score,
            "max_possible_score": max_score
        }
        
        print("\n" + "=" * 70)
        print("*** MICROSERVICE KSE INTEGRATION SUMMARY")
        print("=" * 70)
        
        print(f"Total Services: {summary['total_services']}")
        print(f"Services with KSE Imports: {summary['services_with_kse']}/{summary['total_services']}")
        print(f"Services with Read Operations: {summary['services_with_read']}/{summary['total_services']}")
        print(f"Services with Write Operations: {summary['services_with_write']}/{summary['total_services']}")
        print(f"Services with Trace Operations: {summary['services_with_trace']}/{summary['total_services']}")
        print(f"Average Integration Score: {summary['average_score']:.1f}/10")
        print(f"Overall Score: {summary['total_score']}/{summary['max_possible_score']}")
        
        # Identify services needing attention
        poor_services = [name for name, result in results.items() if result["overall_score"] < 4]
        partial_services = [name for name, result in results.items() if 4 <= result["overall_score"] < 8]
        good_services = [name for name, result in results.items() if result["overall_score"] >= 8]
        
        if poor_services:
            print(f"\n[-] Services needing KSE integration: {', '.join(poor_services)}")
        
        if partial_services:
            print(f"\n[~] Services with partial KSE integration: {', '.join(partial_services)}")
        
        if good_services:
            print(f"\n[+] Services with good KSE integration: {', '.join(good_services)}")
        
        return {
            "summary": summary,
            "services": results,
            "poor_services": poor_services,
            "partial_services": partial_services,
            "good_services": good_services
        }

def main():
    """Run the microservice KSE integration diagnostic"""
    diagnostic = MicroserviceKSEDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    # Return appropriate exit code
    if results["summary"]["average_score"] >= 8:
        print("\n[SUCCESS] All microservices have excellent KSE integration!")
        return 0
    elif results["summary"]["average_score"] >= 6:
        print("\n[WARNING] Most microservices have good KSE integration, some improvements needed")
        return 0
    else:
        print("\n[CRITICAL] Many microservices lack proper KSE integration")
        return 1

if __name__ == "__main__":
    sys.exit(main())