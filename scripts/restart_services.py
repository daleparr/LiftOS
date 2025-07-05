#!/usr/bin/env python3
"""
Script to restart all Lift OS Core services
"""
import subprocess
import time
import sys
import os

def restart_services():
    """Restart all services with updated code"""
    print("Restarting Lift OS Core services...")
    
    services = [
        {"name": "gateway", "port": 8000, "path": "services/gateway"},
        {"name": "auth", "port": 8001, "path": "services/auth"},
        {"name": "memory", "port": 8003, "path": "services/memory"},
        {"name": "registry", "port": 8005, "path": "services/registry"}
    ]
    
    # Kill existing processes
    print("Stopping existing services...")
    for service in services:
        try:
            # Kill processes using the port
            subprocess.run(f"netstat -ano | findstr :{service['port']}", shell=True, capture_output=True)
            subprocess.run(f"taskkill /F /IM python.exe", shell=True, capture_output=True)
        except:
            pass
    
    print("Waiting for services to stop...")
    time.sleep(3)
    
    # Start services
    print("Starting services with updated code...")
    processes = []
    
    for service in services:
        print(f"Starting {service['name']} service on port {service['port']}...")
        try:
            # Change to service directory and start
            cmd = f"cd {service['path']} && python app.py"
            process = subprocess.Popen(cmd, shell=True, cwd=os.getcwd())
            processes.append({"name": service['name'], "process": process})
            time.sleep(2)  # Wait between service starts
        except Exception as e:
            print(f"Failed to start {service['name']}: {e}")
    
    print("All services started. Waiting for initialization...")
    time.sleep(10)
    
    # Test services
    print("Testing service health...")
    import requests
    
    for service in services:
        try:
            response = requests.get(f"http://localhost:{service['port']}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ {service['name']} service is healthy")
            else:
                print(f"✗ {service['name']} service returned {response.status_code}")
        except Exception as e:
            print(f"✗ {service['name']} service is not responding: {e}")
    
    print("\nService restart complete!")
    print("You can now run the production features test again.")

if __name__ == "__main__":
    restart_services()