#!/usr/bin/env python3
"""
Pre-Deployment Validation Script
Comprehensive testing before GitHub push to ensure:
1. Deployment works correctly
2. Frontend is properly connected
3. All documentation is in place
4. System integration is functional
"""

import asyncio
import subprocess
import time
import sys
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

class PreDeploymentValidator:
    """Comprehensive pre-deployment validation system"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.results: List[ValidationResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Service endpoints to test
        self.service_endpoints = {
            "data-ingestion": "http://localhost:8002",
            "streamlit-frontend": "http://localhost:8501"
        }
        
        # Documentation files to check
        self.required_docs = [
            "README.md",
            "shared/kse_sdk/README.md",
            "services/data-ingestion/TIER1_CONNECTORS_README.md",
            "services/data-ingestion/TIER2_CONNECTORS_README.md",
            "services/data-ingestion/TIER3_CONNECTORS_README.md"
        ]
        
        # Critical system files
        self.critical_files = [
            "shared/kse_sdk/diagnostic_tool.py",
            "shared/kse_sdk/microservice_integration_diagnostic.py", 
            "shared/kse_sdk/diagnostics/phase2_diagnostic.py",
            "shared/kse_sdk/client.py",
            "shared/kse_sdk/core/memory.py"
        ]
    
    async def validate_kse_sdk_core(self) -> ValidationResult:
        """Validate KSE-SDK core functionality"""
        start_time = time.time()
        test_name = "KSE-SDK Core Validation"
        
        try:
            # Run KSE-SDK diagnostic
            result = subprocess.run(
                [sys.executable, "shared/kse_sdk/diagnostic_tool.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0 and "8/8 tests passed" in result.stdout
            details = {
                "exit_code": result.returncode,
                "tests_passed": "8/8" if success else "Failed",
                "output_length": len(result.stdout)
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details, 
                                  result.stderr if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_microservice_integration(self) -> ValidationResult:
        """Validate microservice KSE integration"""
        start_time = time.time()
        test_name = "Microservice Integration Validation"
        
        try:
            result = subprocess.run(
                [sys.executable, "shared/kse_sdk/microservice_integration_diagnostic.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0 and "16/16" in result.stdout
            details = {
                "exit_code": result.returncode,
                "services_integrated": "16/16" if success else "Failed",
                "kse_coverage": "100%" if success else "Incomplete"
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  result.stderr if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_phase2_intelligence(self) -> ValidationResult:
        """Validate Phase 2 advanced intelligence flows"""
        start_time = time.time()
        test_name = "Phase 2 Intelligence Flow Validation"
        
        try:
            result = subprocess.run(
                [sys.executable, "shared/kse_sdk/diagnostics/phase2_diagnostic.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            success = result.returncode == 0 and "8/8 tests passed" in result.stdout
            details = {
                "exit_code": result.returncode,
                "intelligence_tests": "8/8" if success else "Failed",
                "success_rate": "100%" if success else "Incomplete"
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  result.stderr if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_frontend_connectivity(self) -> ValidationResult:
        """Validate frontend is running and accessible"""
        start_time = time.time()
        test_name = "Frontend Connectivity Validation"
        
        try:
            # Check if Streamlit is running
            response = requests.get(
                "http://localhost:8501",
                timeout=10,
                allow_redirects=True
            )
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_size": len(response.content),
                "streamlit_running": success,
                "content_type": response.headers.get("content-type", "unknown")
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  f"HTTP {response.status_code}" if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_data_ingestion_service(self) -> ValidationResult:
        """Validate data ingestion service is running"""
        start_time = time.time()
        test_name = "Data Ingestion Service Validation"
        
        try:
            # Check if data ingestion service is running
            response = requests.get(
                "http://localhost:8006/health",
                timeout=10
            )
            
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "service_running": success,
                "health_check": "passed" if success else "failed"
            }
            
            if success:
                try:
                    health_data = response.json()
                    details["health_data"] = health_data
                except:
                    pass
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  f"HTTP {response.status_code}" if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_documentation(self) -> ValidationResult:
        """Validate all required documentation exists"""
        start_time = time.time()
        test_name = "Documentation Validation"
        
        try:
            missing_docs = []
            existing_docs = []
            
            for doc_path in self.required_docs:
                file_path = self.base_dir / doc_path
                if file_path.exists():
                    existing_docs.append(doc_path)
                else:
                    missing_docs.append(doc_path)
            
            success = len(missing_docs) == 0
            details = {
                "total_docs": len(self.required_docs),
                "existing_docs": len(existing_docs),
                "missing_docs": len(missing_docs),
                "missing_list": missing_docs,
                "coverage_percent": (len(existing_docs) / len(self.required_docs)) * 100
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  f"Missing docs: {missing_docs}" if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_critical_files(self) -> ValidationResult:
        """Validate all critical system files exist"""
        start_time = time.time()
        test_name = "Critical Files Validation"
        
        try:
            missing_files = []
            existing_files = []
            
            for file_path in self.critical_files:
                full_path = self.base_dir / file_path
                if full_path.exists():
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            success = len(missing_files) == 0
            details = {
                "total_files": len(self.critical_files),
                "existing_files": len(existing_files),
                "missing_files": len(missing_files),
                "missing_list": missing_files
            }
            
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details,
                                  f"Missing files: {missing_files}" if not success else None)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def validate_git_status(self) -> ValidationResult:
        """Validate git repository status"""
        start_time = time.time()
        test_name = "Git Repository Validation"
        
        try:
            # Check git status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if we're in a git repository
            if status_result.returncode != 0:
                duration = time.time() - start_time
                return ValidationResult(test_name, False, duration, {}, 
                                      "Not in a git repository")
            
            modified_files = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            
            details = {
                "current_branch": current_branch,
                "modified_files_count": len(modified_files),
                "has_uncommitted_changes": len(modified_files) > 0,
                "git_ready": len(modified_files) == 0
            }
            
            success = True  # Git validation is informational
            duration = time.time() - start_time
            return ValidationResult(test_name, success, duration, details)
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name, False, duration, {}, str(e))
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("="*80)
        print("PRE-DEPLOYMENT VALIDATION SUITE")
        print("="*80)
        print("Preparing for GitHub push - validating system readiness...")
        print()
        
        # Run all validation tests
        validations = [
            ("KSE-SDK Core", self.validate_kse_sdk_core()),
            ("Microservice Integration", self.validate_microservice_integration()),
            ("Phase 2 Intelligence", self.validate_phase2_intelligence()),
            ("Frontend Connectivity", self.validate_frontend_connectivity()),
            ("Data Ingestion Service", self.validate_data_ingestion_service()),
            ("Documentation", self.validate_documentation()),
            ("Critical Files", self.validate_critical_files()),
            ("Git Repository", self.validate_git_status())
        ]
        
        for test_name, validation_coro in validations:
            print(f"[*] Running {test_name}...")
            result = await validation_coro
            self.results.append(result)
            
            status = "[PASS]" if result.success else "[FAIL]"
            print(f"    {status} ({result.duration:.2f}s)")
            if not result.success and result.error:
                print(f"    Error: {result.error}")
            print()
        
        # Generate summary
        passed_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "deployment_ready": success_rate >= 85,  # 85% threshold for deployment
            "results": [
                {
                    "test": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Deployment Ready: {'[YES]' if summary['deployment_ready'] else '[NO]'}")
        print()
        
        if summary['deployment_ready']:
            print("[SUCCESS] SYSTEM IS READY FOR GITHUB PUSH!")
        else:
            print("[WARNING] SYSTEM NEEDS ATTENTION BEFORE GITHUB PUSH")
            print("\nFailed Tests:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.error}")
        
        print("="*80)
        return summary

async def main():
    """Main validation entry point"""
    validator = PreDeploymentValidator()
    summary = await validator.run_all_validations()
    
    # Save detailed results
    with open("pre_deployment_validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: pre_deployment_validation_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if summary['deployment_ready'] else 1)

if __name__ == "__main__":
    asyncio.run(main())