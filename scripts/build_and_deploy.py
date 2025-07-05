#!/usr/bin/env python3
"""
Build and Deploy LiftOS System
Builds all required Docker images locally, then deploys the system
"""

import subprocess
import time
import requests
import json
import sys
from datetime import datetime

def run_command(cmd, timeout=300):
    """Run a command and return success status"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def build_all_images():
    """Build all required Docker images"""
    print("\nBuilding all Docker images...")
    
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
    ]
    
    for cmd in build_commands:
        if not run_command(cmd, timeout=600):
            return False
    
    return True

def deploy_system():
    """Deploy the system using docker-compose"""
    print("\nDeploying LiftOS system...")
    
    try:
        # Stop existing containers
        print("Stopping existing containers...")
        subprocess.run(['docker-compose', '-f', 'docker-compose.production.yml', 'down'], 
                     capture_output=True, timeout=60)
        
        # Start the system
        print("Starting system...")
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.production.yml', 'up', '-d'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"ERROR: Deployment failed: {result.stderr}")
            return False
        
        print("SUCCESS: System deployment initiated")
        return True
        
    except Exception as e:
        print(f"ERROR: Deployment exception: {e}")
        return False

def wait_for_services():
    """Wait for services to become healthy"""
    print("\nWaiting for services to start...")
    
    services = {
        'gateway': 'http://localhost:8000/health',
        'auth': 'http://localhost:8001/health',
        'registry': 'http://localhost:8005/health',
        'surfacing': 'http://localhost:8007/health',
        'causal': 'http://localhost:8008/health',
        'llm': 'http://localhost:8009/health'
    }
    
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        all_healthy = True
        
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    all_healthy = False
                    break
            except:
                all_healthy = False
                break
        
        if all_healthy:
            print("SUCCESS: All services are healthy!")
            return True
        
        print("INFO: Waiting for services...")
        time.sleep(10)
    
    print("WARNING: Timeout waiting for all services")
    return False

def test_services():
    """Test core services"""
    print("\nTesting services...")
    
    services = {
        'gateway': 'http://localhost:8000/health',
        'auth': 'http://localhost:8001/health',
        'registry': 'http://localhost:8005/health',
        'surfacing': 'http://localhost:8007/health',
        'causal': 'http://localhost:8008/health',
        'llm': 'http://localhost:8009/health'
    }
    
    results = {}
    
    for service, url in services.items():
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"SUCCESS: {service}: {response_time:.3f}s")
                results[service] = {'status': 'OK', 'time': response_time}
            else:
                print(f"ERROR: {service}: HTTP {response.status_code}")
                results[service] = {'status': 'FAIL', 'code': response.status_code}
                
        except Exception as e:
            print(f"ERROR: {service}: {e}")
            results[service] = {'status': 'FAIL', 'error': str(e)}
    
    return results

def main():
    """Main build and deploy function"""
    print("LiftOS Build and Deploy")
    print("=" * 40)
    
    # Step 1: Build all images
    if not build_all_images():
        print("ERROR: Image building failed")
        return False
    
    # Step 2: Deploy system
    if not deploy_system():
        print("ERROR: System deployment failed")
        return False
    
    # Step 3: Wait for services
    if not wait_for_services():
        print("WARNING: Not all services became healthy")
    
    # Step 4: Test services
    results = test_services()
    
    # Step 5: Show results
    print("\n" + "=" * 40)
    print("DEPLOYMENT RESULTS")
    print("=" * 40)
    
    success_count = 0
    total_count = len(results)
    
    for service, result in results.items():
        status = result.get('status', 'UNKNOWN')
        if status == 'OK':
            success_count += 1
            time_info = f" ({result.get('time', 0):.3f}s)"
            print(f"OK   {service}{time_info}")
        else:
            print(f"FAIL {service}")
    
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    # Show container status
    print("\nContainer Status:")
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not get container status: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"build_deploy_report_{timestamp}.json"
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'success_rate': success_rate,
        'results': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    return success_rate > 50

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nSUCCESS: Build and deployment completed!")
        sys.exit(0)
    else:
        print("\nWARNING: Build or deployment had issues")
        sys.exit(1)