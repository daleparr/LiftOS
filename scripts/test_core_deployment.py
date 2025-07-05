#!/usr/bin/env python3
"""
Test Core Services Deployment
Deploy and test just the core LiftOS services
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

def deploy_core_services():
    """Deploy core services using simplified docker-compose"""
    print("\nDeploying core services...")
    
    try:
        # Stop existing containers
        print("Stopping existing containers...")
        subprocess.run(['docker-compose', '-f', 'docker-compose.core.yml', 'down'], 
                     capture_output=True, timeout=60)
        
        # Start core services
        print("Starting core services...")
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose.core.yml', 'up', '-d'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"ERROR: Deployment failed: {result.stderr}")
            return False
        
        print("SUCCESS: Core services deployment initiated")
        return True
        
    except Exception as e:
        print(f"ERROR: Deployment exception: {e}")
        return False

def test_core_services():
    """Test core services"""
    print("\nTesting core services...")
    
    services = {
        'gateway': 'http://localhost:8000/health',
        'auth': 'http://localhost:8001/health',
        'registry': 'http://localhost:8005/health'
    }
    
    results = {}
    max_wait = 180  # 3 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        all_ready = True
        
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"SUCCESS: {service} is healthy")
                    results[service] = {'status': 'OK', 'response_time': response.elapsed.total_seconds()}
                else:
                    print(f"INFO: {service} not ready yet (HTTP {response.status_code})")
                    all_ready = False
                    results[service] = {'status': 'PENDING', 'code': response.status_code}
            except Exception as e:
                print(f"INFO: {service} not ready yet ({str(e)[:50]})")
                all_ready = False
                results[service] = {'status': 'PENDING', 'error': str(e)[:50]}
        
        if all_ready:
            print("\nSUCCESS: All core services are healthy!")
            break
        
        print("INFO: Waiting for services to become ready...")
        time.sleep(10)
    
    return results

def main():
    """Main test function"""
    print("LiftOS Core Services Test")
    print("=" * 40)
    
    # Step 1: Deploy core services
    if not deploy_core_services():
        print("ERROR: Core services deployment failed")
        return False
    
    # Step 2: Wait and test services
    print("\nWaiting for services to start...")
    time.sleep(30)  # Give services time to start
    
    results = test_core_services()
    
    # Step 3: Show results
    print("\n" + "=" * 40)
    print("CORE SERVICES TEST RESULTS")
    print("=" * 40)
    
    success_count = 0
    total_count = len(results)
    
    for service, result in results.items():
        status = result.get('status', 'UNKNOWN')
        if status == 'OK':
            success_count += 1
            time_info = f" ({result.get('response_time', 0):.3f}s)"
            print(f"OK   {service}{time_info}")
        else:
            print(f"FAIL {service} - {status}")
    
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nCore Services Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    # Step 4: Show container status
    print("\nContainer Status:")
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=lift-'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not get container status: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"core_services_test_{timestamp}.json"
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'success_rate': success_rate,
        'results': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    return success_rate > 66  # At least 2 out of 3 services

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nSUCCESS: Core services test completed!")
        sys.exit(0)
    else:
        print("\nWARNING: Core services test had issues")
        sys.exit(1)