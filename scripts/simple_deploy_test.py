#!/usr/bin/env python3
"""
Simple LiftOS Deployment and Testing Script
"""

import subprocess
import time
import requests
import json
import sys
from datetime import datetime

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception as e:
        print(f"Docker check failed: {e}")
        return False

def wait_for_docker(max_wait=300):
    """Wait for Docker to become available"""
    print("Waiting for Docker Desktop to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if check_docker():
            print("SUCCESS: Docker is ready!")
            return True
        print("INFO: Docker not ready yet, waiting...")
        time.sleep(10)
    
    print("ERROR: Docker failed to become ready within timeout")
    return False

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
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"ERROR: Deployment failed: {result.stderr}")
            return False
        
        print("SUCCESS: System deployment initiated")
        return True
        
    except Exception as e:
        print(f"ERROR: Deployment exception: {e}")
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
    """Main deployment and test function"""
    print("Starting LiftOS Deployment and Test")
    print("=" * 40)
    
    # Step 1: Wait for Docker
    if not wait_for_docker():
        print("ERROR: Docker is not available")
        return False
    
    # Step 2: Deploy system
    if not deploy_system():
        print("ERROR: System deployment failed")
        return False
    
    # Step 3: Wait for services to start
    print("\nWaiting for services to start (60 seconds)...")
    time.sleep(60)
    
    # Step 4: Test services
    results = test_services()
    
    # Step 5: Show results
    print("\n" + "=" * 40)
    print("DEPLOYMENT TEST RESULTS")
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
    
    # Step 6: Show container status
    print("\nContainer Status:")
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not get container status: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"deployment_test_{timestamp}.json"
    
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
        print("\nSUCCESS: Deployment completed!")
        sys.exit(0)
    else:
        print("\nWARNING: Deployment had issues")
        sys.exit(1)