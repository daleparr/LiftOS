#!/usr/bin/env python3
"""
LiftOS Microservices Stability and Robustness Analysis

This script performs comprehensive stability and robustness checks for the three
integrated microservices: Surfacing, Causal AI, and LLM.
"""

import json
import time
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import statistics

class StabilityAnalyzer:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "microservices": ["surfacing", "causal", "llm"],
            "tests": {},
            "summary": {}
        }
        
        # Service endpoints for testing
        self.services = {
            "core": {
                "gateway": "http://localhost:8000",
                "auth": "http://localhost:8001", 
                "memory": "http://localhost:8003",
                "registry": "http://localhost:8005"
            },
            "microservices": {
                "surfacing_service": "http://localhost:3002",
                "surfacing_module": "http://localhost:8007",
                "causal_service": "http://localhost:3003", 
                "causal_module": "http://localhost:8008",
                "llm_service": "http://localhost:3004",
                "llm_module": "http://localhost:8009"
            }
        }

    def log_result(self, test_name: str, status: str, details: Dict[str, Any]):
        """Log test results with timestamp"""
        self.results["tests"][test_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        print(f"[{status}] {test_name}")
        if details.get("message"):
            print(f"    {details['message']}")

    def test_service_availability(self) -> Dict[str, Any]:
        """Test basic availability of all services"""
        print("\n=== SERVICE AVAILABILITY ANALYSIS ===")
        availability = {}
        
        for category, services in self.services.items():
            availability[category] = {}
            for service_name, url in services.items():
                try:
                    start_time = time.time()
                    response = requests.get(f"{url}/health", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    
                    availability[category][service_name] = {
                        "available": response.status_code == 200,
                        "response_time_ms": round(response_time, 2),
                        "status_code": response.status_code
                    }
                    
                    status = "PASS" if response.status_code == 200 else "FAIL"
                    self.log_result(f"Availability: {service_name}", status, {
                        "message": f"Response time: {response_time:.2f}ms",
                        "response_time": response_time,
                        "status_code": response.status_code
                    })
                    
                except Exception as e:
                    availability[category][service_name] = {
                        "available": False,
                        "error": str(e)
                    }
                    self.log_result(f"Availability: {service_name}", "FAIL", {
                        "message": f"Connection failed: {str(e)}"
                    })
        
        return availability

    def test_response_time_consistency(self) -> Dict[str, Any]:
        """Test response time consistency over multiple requests"""
        print("\n=== RESPONSE TIME CONSISTENCY ANALYSIS ===")
        consistency_results = {}
        
        # Test only available services
        available_services = []
        for category, services in self.services.items():
            for service_name, url in services.items():
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        available_services.append((service_name, url))
                except:
                    continue
        
        for service_name, url in available_services:
            response_times = []
            errors = 0
            
            print(f"Testing {service_name} consistency...")
            for i in range(10):  # 10 requests per service
                try:
                    start_time = time.time()
                    response = requests.get(f"{url}/health", timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        response_times.append(response_time)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
                
                time.sleep(0.1)  # Small delay between requests
            
            if response_times:
                avg_time = statistics.mean(response_times)
                std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
                min_time = min(response_times)
                max_time = max(response_times)
                
                consistency_results[service_name] = {
                    "average_ms": round(avg_time, 2),
                    "std_deviation": round(std_dev, 2),
                    "min_ms": round(min_time, 2),
                    "max_ms": round(max_time, 2),
                    "error_rate": errors / 10,
                    "consistency_score": round(100 - (std_dev / avg_time * 100), 2) if avg_time > 0 else 0
                }
                
                status = "PASS" if std_dev < avg_time * 0.3 and errors == 0 else "WARN"
                self.log_result(f"Consistency: {service_name}", status, {
                    "message": f"Avg: {avg_time:.2f}ms, StdDev: {std_dev:.2f}ms, Errors: {errors}/10",
                    "metrics": consistency_results[service_name]
                })
            else:
                consistency_results[service_name] = {"error": "No successful responses"}
                self.log_result(f"Consistency: {service_name}", "FAIL", {
                    "message": "No successful responses during consistency test"
                })
        
        return consistency_results

    def test_configuration_integrity(self) -> Dict[str, Any]:
        """Test configuration integrity across microservices"""
        print("\n=== CONFIGURATION INTEGRITY ANALYSIS ===")
        config_results = {}
        
        # Load module configurations
        modules = ["surfacing", "causal", "llm"]
        for module in modules:
            try:
                with open(f"modules/{module}/module.json", "r") as f:
                    config = json.load(f)
                
                # Check required fields
                required_fields = ["name", "version", "capabilities", "endpoints", "dependencies"]
                missing_fields = [field for field in required_fields if field not in config]
                
                # Check endpoint consistency
                endpoint_count = len(config.get("endpoints", {}))
                capability_count = len(config.get("capabilities", []))
                
                config_results[module] = {
                    "valid": len(missing_fields) == 0,
                    "missing_fields": missing_fields,
                    "endpoint_count": endpoint_count,
                    "capability_count": capability_count,
                    "has_health_check": "health_check" in config,
                    "has_resource_requirements": "resource_requirements" in config
                }
                
                status = "PASS" if len(missing_fields) == 0 else "FAIL"
                self.log_result(f"Config Integrity: {module}", status, {
                    "message": f"Endpoints: {endpoint_count}, Capabilities: {capability_count}",
                    "missing_fields": missing_fields
                })
                
            except Exception as e:
                config_results[module] = {"error": str(e)}
                self.log_result(f"Config Integrity: {module}", "FAIL", {
                    "message": f"Failed to load config: {str(e)}"
                })
        
        return config_results

    def test_dependency_chain_stability(self) -> Dict[str, Any]:
        """Test stability of dependency chains"""
        print("\n=== DEPENDENCY CHAIN STABILITY ANALYSIS ===")
        dependency_results = {}
        
        # Define dependency chains for each microservice
        dependency_chains = {
            "surfacing": [
                "auth", "memory", "registry", "surfacing-service", "surfacing-module"
            ],
            "causal": [
                "auth", "memory", "registry", "causal-service", "causal-module"  
            ],
            "llm": [
                "auth", "memory", "registry", "llm-service", "llm-module"
            ]
        }
        
        for microservice, chain in dependency_chains.items():
            chain_health = []
            for service in chain:
                # Map service names to URLs
                service_url = None
                if service in ["auth", "memory", "registry"]:
                    service_url = self.services["core"].get(service)
                else:
                    # Map to microservice URLs
                    service_map = {
                        "surfacing-service": "surfacing_service",
                        "surfacing-module": "surfacing_module", 
                        "causal-service": "causal_service",
                        "causal-module": "causal_module",
                        "llm-service": "llm_service",
                        "llm-module": "llm_module"
                    }
                    mapped_name = service_map.get(service)
                    if mapped_name:
                        service_url = self.services["microservices"].get(mapped_name)
                
                if service_url:
                    try:
                        response = requests.get(f"{service_url}/health", timeout=3)
                        chain_health.append({
                            "service": service,
                            "healthy": response.status_code == 200,
                            "status_code": response.status_code
                        })
                    except Exception as e:
                        chain_health.append({
                            "service": service,
                            "healthy": False,
                            "error": str(e)
                        })
                else:
                    chain_health.append({
                        "service": service,
                        "healthy": False,
                        "error": "Service URL not found"
                    })
            
            healthy_count = sum(1 for item in chain_health if item["healthy"])
            total_count = len(chain_health)
            health_percentage = (healthy_count / total_count) * 100 if total_count > 0 else 0
            
            dependency_results[microservice] = {
                "chain_health": chain_health,
                "healthy_services": healthy_count,
                "total_services": total_count,
                "health_percentage": round(health_percentage, 2),
                "fully_operational": healthy_count == total_count
            }
            
            status = "PASS" if health_percentage == 100 else "WARN" if health_percentage >= 60 else "FAIL"
            self.log_result(f"Dependency Chain: {microservice}", status, {
                "message": f"{healthy_count}/{total_count} services healthy ({health_percentage:.1f}%)",
                "health_percentage": health_percentage
            })
        
        return dependency_results

    def test_error_handling_robustness(self) -> Dict[str, Any]:
        """Test error handling and robustness"""
        print("\n=== ERROR HANDLING ROBUSTNESS ANALYSIS ===")
        robustness_results = {}
        
        # Test invalid endpoints
        test_cases = [
            {"endpoint": "/invalid", "expected_status": [404, 405]},
            {"endpoint": "/health/../admin", "expected_status": [400, 404, 403]},
            {"endpoint": "/api/v1/nonexistent", "expected_status": [404, 405]}
        ]
        
        available_services = []
        for category, services in self.services.items():
            for service_name, url in services.items():
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        available_services.append((service_name, url))
                except:
                    continue
        
        for service_name, base_url in available_services:
            service_robustness = {"tests": [], "score": 0}
            
            for test_case in test_cases:
                try:
                    response = requests.get(f"{base_url}{test_case['endpoint']}", timeout=5)
                    expected_statuses = test_case["expected_status"]
                    
                    test_passed = response.status_code in expected_statuses
                    service_robustness["tests"].append({
                        "endpoint": test_case["endpoint"],
                        "status_code": response.status_code,
                        "expected": expected_statuses,
                        "passed": test_passed
                    })
                    
                    if test_passed:
                        service_robustness["score"] += 1
                        
                except Exception as e:
                    service_robustness["tests"].append({
                        "endpoint": test_case["endpoint"],
                        "error": str(e),
                        "passed": False
                    })
            
            total_tests = len(test_cases)
            robustness_percentage = (service_robustness["score"] / total_tests) * 100
            service_robustness["robustness_percentage"] = robustness_percentage
            
            robustness_results[service_name] = service_robustness
            
            status = "PASS" if robustness_percentage >= 80 else "WARN" if robustness_percentage >= 60 else "FAIL"
            self.log_result(f"Error Handling: {service_name}", status, {
                "message": f"Robustness: {robustness_percentage:.1f}% ({service_robustness['score']}/{total_tests})",
                "robustness_percentage": robustness_percentage
            })
        
        return robustness_results

    def generate_stability_report(self) -> Dict[str, Any]:
        """Generate comprehensive stability report"""
        print("\n=== GENERATING STABILITY REPORT ===")
        
        # Run all tests
        availability = self.test_service_availability()
        consistency = self.test_response_time_consistency()
        config_integrity = self.test_configuration_integrity()
        dependency_stability = self.test_dependency_chain_stability()
        error_handling = self.test_error_handling_robustness()
        
        # Calculate overall scores
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test["status"] == "PASS")
        warned_tests = sum(1 for test in self.results["tests"].values() if test["status"] == "WARN")
        failed_tests = sum(1 for test in self.results["tests"].values() if test["status"] == "FAIL")
        
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate summary
        summary = {
            "overall_stability_score": round(overall_score, 2),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "warned_tests": warned_tests,
            "failed_tests": failed_tests,
            "test_categories": {
                "availability": availability,
                "consistency": consistency,
                "configuration": config_integrity,
                "dependencies": dependency_stability,
                "error_handling": error_handling
            },
            "recommendations": self.generate_recommendations()
        }
        
        self.results["summary"] = summary
        return self.results

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for failed services
        failed_tests = [name for name, result in self.results["tests"].items() 
                       if result["status"] == "FAIL"]
        
        if any("Availability" in test for test in failed_tests):
            recommendations.append("Deploy missing microservices using setup scripts")
            recommendations.append("Verify Docker containers are running and healthy")
        
        if any("Consistency" in test for test in failed_tests):
            recommendations.append("Investigate response time variations and optimize service performance")
            recommendations.append("Consider implementing connection pooling and caching")
        
        if any("Config Integrity" in test for test in failed_tests):
            recommendations.append("Review and fix module configuration files")
            recommendations.append("Ensure all required configuration fields are present")
        
        if any("Dependency Chain" in test for test in failed_tests):
            recommendations.append("Fix dependency chain issues by ensuring all services are healthy")
            recommendations.append("Implement proper service startup ordering")
        
        if any("Error Handling" in test for test in failed_tests):
            recommendations.append("Improve error handling and HTTP status code responses")
            recommendations.append("Implement proper input validation and security measures")
        
        if not recommendations:
            recommendations.append("All stability tests passed - system is robust and ready for production")
        
        return recommendations

def main():
    """Main execution function"""
    print("=" * 60)
    print("LiftOS Microservices Stability & Robustness Analysis")
    print("=" * 60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyzer = StabilityAnalyzer()
    results = analyzer.generate_stability_report()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Overall Stability Score: {summary['overall_stability_score']:.1f}%")
    print(f"Tests: {summary['passed_tests']} passed, {summary['warned_tests']} warnings, {summary['failed_tests']} failed")
    print()
    
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(summary["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"stability_report_{timestamp}.json"
    
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return summary['overall_stability_score'] >= 70  # Return True if stable

if __name__ == "__main__":
    main()