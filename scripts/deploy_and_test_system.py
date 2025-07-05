#!/usr/bin/env python3
"""
LiftOS Complete System Deployment and Testing Script
Deploys all services and microservices, then runs comprehensive stability tests
"""

import subprocess
import time
import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class LiftOSDeploymentTester:
    def __init__(self):
        self.services = {
            # Core LiftOS Services
            'gateway': {'port': 8000, 'url': 'http://localhost:8000'},
            'auth': {'port': 8001, 'url': 'http://localhost:8001'},
            'billing': {'port': 8002, 'url': 'http://localhost:8002'},
            'memory': {'port': 8003, 'url': 'http://localhost:8003'},
            'observability': {'port': 8004, 'url': 'http://localhost:8004'},
            'registry': {'port': 8005, 'url': 'http://localhost:8005'},
            
            # External Microservices
            'surfacing-service': {'port': 3002, 'url': 'http://localhost:3002'},
            'causal-service': {'port': 3003, 'url': 'http://localhost:3003'},
            'llm-service': {'port': 3004, 'url': 'http://localhost:3004'},
            
            # LiftOS Module Wrappers
            'surfacing': {'port': 8007, 'url': 'http://localhost:8007'},
            'causal': {'port': 8008, 'url': 'http://localhost:8008'},
            'llm': {'port': 8009, 'url': 'http://localhost:8009'},
            
            # Infrastructure
            'postgres': {'port': 5432, 'url': 'http://localhost:5432'},
            'redis': {'port': 6379, 'url': 'http://localhost:6379'},
            'prometheus': {'port': 9090, 'url': 'http://localhost:9090'},
            'grafana': {'port': 3000, 'url': 'http://localhost:3000'},
        }
        
        self.deployment_results = {}
        self.test_results = {}
        
    def check_docker_status(self) -> bool:
        """Check if Docker is running and accessible"""
        try:
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            print(f"Docker check failed: {e}")
            return False
    
    def wait_for_docker(self, max_wait: int = 300) -> bool:
        """Wait for Docker to become available"""
        print("Waiting for Docker Desktop to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.check_docker_status():
                print("SUCCESS: Docker is ready!")
                return True
            print("INFO: Docker not ready yet, waiting...")
            time.sleep(10)
        
        print("ERROR: Docker failed to become ready within timeout")
        return False
    
    def build_images(self) -> bool:
        """Build all required Docker images"""
        print("\nBuilding Docker images...")
        
        build_commands = [
            # Core services
            "docker build -t lift/gateway-service:latest -f services/gateway/Dockerfile .",
            "docker build -t lift/auth-service:latest -f services/auth/Dockerfile .",
            "docker build -t lift/billing-service:latest -f services/billing/Dockerfile .",
            "docker build -t lift/memory-service:latest -f services/memory/Dockerfile .",
            "docker build -t lift/observability-service:latest -f services/observability/Dockerfile .",
            "docker build -t lift/registry-service:latest -f services/registry/Dockerfile .",
            
            # Module wrappers
            "docker build -t lift/surfacing-module:latest -f modules/surfacing/Dockerfile .",
            "docker build -t lift/causal-module:latest -f modules/causal/Dockerfile .",
            "docker build -t lift/llm-module:latest -f modules/llm/Dockerfile .",
            
            # LLM service (local build)
            "docker build -t lift/llm-service:latest -f Dockerfile.llm-service .",
        ]
        
        for cmd in build_commands:
            print(f"Building: {cmd}")
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"ERROR: Build failed: {cmd}")
                    print(f"Error: {result.stderr}")
                    return False
                print(f"✅ Built successfully")
            except Exception as e:
                print(f"ERROR: Build exception: {e}")
                return False
        
        return True
    
    def deploy_system(self) -> bool:
        """Deploy the complete system using docker-compose"""
        print("\nDeploying LiftOS system...")
        
        try:
            # Stop any existing containers
            print("Stopping existing containers...")
            subprocess.run(['docker-compose', '-f', 'docker-compose.production.yml', 'down'], 
                         capture_output=True, timeout=60)
            
            # Start the system
            print("Starting system...")
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.production.yml', 'up', '-d'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"ERROR: Deployment failed: {result.stderr}")
                return False
            
            print("SUCCESS: System deployment initiated")
            return True
            
        except Exception as e:
            print(f"ERROR: Deployment exception: {e}")
            return False
    
    def wait_for_services(self, timeout: int = 300) -> Dict[str, bool]:
        """Wait for all services to become healthy"""
        print("\nWaiting for services to become healthy...")
        
        start_time = time.time()
        service_status = {name: False for name in self.services.keys()}
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service_name, config in self.services.items():
                if service_status[service_name]:
                    continue
                
                try:
                    # Check health endpoint
                    health_url = f"{config['url']}/health"
                    response = requests.get(health_url, timeout=5)
                    
                    if response.status_code == 200:
                        service_status[service_name] = True
                        print(f"SUCCESS: {service_name} is healthy")
                    else:
                        all_healthy = False
                        
                except Exception:
                    all_healthy = False
            
            if all_healthy:
                print("🎉 All services are healthy!")
                return service_status
            
            time.sleep(10)
        
        print("WARNING: Timeout waiting for all services to become healthy")
        return service_status
    
    def run_stability_tests(self) -> Dict[str, any]:
        """Run comprehensive stability tests"""
        print("\nRunning stability tests...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'service_availability': {},
            'response_times': {},
            'integration_tests': {},
            'load_tests': {},
            'overall_score': 0
        }
        
        # Test service availability
        print("Testing service availability...")
        for service_name, config in self.services.items():
            try:
                start_time = time.time()
                response = requests.get(f"{config['url']}/health", timeout=10)
                response_time = time.time() - start_time
                
                test_results['service_availability'][service_name] = {
                    'status': response.status_code == 200,
                    'response_time': response_time,
                    'status_code': response.status_code
                }
                
                if response.status_code == 200:
                    print(f"SUCCESS: {service_name}: {response_time:.3f}s")
                else:
                    print(f"ERROR: {service_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                test_results['service_availability'][service_name] = {
                    'status': False,
                    'error': str(e)
                }
                print(f"ERROR: {service_name}: {e}")
        
        # Test microservice integrations
        print("\nTesting microservice integrations...")
        integration_tests = [
            ('surfacing', '/api/v1/surface', {'query': 'test'}),
            ('causal', '/api/v1/analyze', {'data': 'test'}),
            ('llm', '/api/v1/generate', {'prompt': 'test'})
        ]
        
        for service, endpoint, payload in integration_tests:
            try:
                url = f"http://localhost:{self.services[service]['port']}{endpoint}"
                response = requests.post(url, json=payload, timeout=15)
                
                test_results['integration_tests'][service] = {
                    'status': response.status_code in [200, 201],
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code
                }
                
                if response.status_code in [200, 201]:
                    print(f"SUCCESS: {service} integration: {response.elapsed.total_seconds():.3f}s")
                else:
                    print(f"ERROR: {service} integration: HTTP {response.status_code}")
                    
            except Exception as e:
                test_results['integration_tests'][service] = {
                    'status': False,
                    'error': str(e)
                }
                print(f"ERROR: {service} integration: {e}")
        
        # Calculate overall score
        total_services = len(self.services)
        healthy_services = sum(1 for result in test_results['service_availability'].values() 
                             if result.get('status', False))
        
        total_integrations = len(integration_tests)
        working_integrations = sum(1 for result in test_results['integration_tests'].values() 
                                 if result.get('status', False))
        
        availability_score = (healthy_services / total_services) * 100
        integration_score = (working_integrations / total_integrations) * 100 if total_integrations > 0 else 100
        
        test_results['overall_score'] = (availability_score + integration_score) / 2
        test_results['summary'] = {
            'healthy_services': f"{healthy_services}/{total_services}",
            'working_integrations': f"{working_integrations}/{total_integrations}",
            'availability_score': availability_score,
            'integration_score': integration_score
        }
        
        return test_results
    
    def generate_report(self, test_results: Dict[str, any]) -> str:
        """Generate a comprehensive deployment and test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"deployment_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate summary
        summary = f"""
🎯 LiftOS Deployment and Test Report
=====================================
Timestamp: {test_results['timestamp']}
Overall Score: {test_results['overall_score']:.1f}%

📊 Summary:
- Services: {test_results['summary']['healthy_services']}
- Integrations: {test_results['summary']['working_integrations']}
- Availability: {test_results['summary']['availability_score']:.1f}%
- Integration: {test_results['summary']['integration_score']:.1f}%

📋 Service Status:
"""
        
        for service, result in test_results['service_availability'].items():
            status = "✅" if result.get('status', False) else "❌"
            time_info = f" ({result.get('response_time', 0):.3f}s)" if result.get('response_time') else ""
            summary += f"{status} {service}{time_info}\n"
        
        summary += f"\n📄 Full report saved to: {report_file}\n"
        
        return summary
    
    def run_complete_deployment_test(self):
        """Run the complete deployment and testing process"""
        print("Starting LiftOS Complete Deployment and Test")
        print("=" * 50)
        
        # Step 1: Wait for Docker
        if not self.wait_for_docker():
            print("ERROR: Docker is not available. Please start Docker Desktop.")
            return False
        
        # Step 2: Build images
        if not self.build_images():
            print("ERROR: Image building failed.")
            return False
        
        # Step 3: Deploy system
        if not self.deploy_system():
            print("ERROR: System deployment failed.")
            return False
        
        # Step 4: Wait for services
        print("\nWaiting for services to start (this may take several minutes)...")
        time.sleep(60)  # Give services time to start
        
        service_status = self.wait_for_services()
        
        # Step 5: Run tests
        test_results = self.run_stability_tests()
        
        # Step 6: Generate report
        summary = self.generate_report(test_results)
        print(summary)
        
        # Step 7: Show container status
        print("\nContainer Status:")
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Could not get container status: {e}")
        
        return test_results['overall_score'] > 75

if __name__ == "__main__":
    tester = LiftOSDeploymentTester()
    success = tester.run_complete_deployment_test()
    
    if success:
        print("\nSUCCESS: Deployment and testing completed successfully!")
        sys.exit(0)
    else:
        print("\nWARNING: Deployment or testing encountered issues.")
        sys.exit(1)