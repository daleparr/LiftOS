#!/usr/bin/env python3
"""
Local development server startup script
Starts all Lift OS Core services for local testing without Docker
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.services = [
            {
                'name': 'gateway',
                'port': 8000,
                'path': 'services/gateway',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000']
            },
            {
                'name': 'auth',
                'port': 8001,
                'path': 'services/auth',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8001']
            },
            {
                'name': 'memory',
                'port': 8002,
                'path': 'services/memory',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8002']
            },
            {
                'name': 'registry',
                'port': 8003,
                'path': 'services/registry',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8003']
            },
            {
                'name': 'billing',
                'port': 8004,
                'path': 'services/billing',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8004']
            },
            {
                'name': 'observability',
                'port': 8005,
                'path': 'services/observability',
                'cmd': ['python', '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8005']
            }
        ]
        
    def install_dependencies(self):
        """Install dependencies for all services"""
        print("Installing dependencies for all services...")
        
        for service in self.services:
            service_path = project_root / service['path']
            requirements_path = service_path / 'requirements.txt'
            
            if requirements_path.exists():
                print(f"Installing dependencies for {service['name']}...")
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)
                    ], check=True, cwd=service_path)
                    print(f"✓ Dependencies installed for {service['name']}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to install dependencies for {service['name']}: {e}")
                    return False
        
        return True
    
    def start_service(self, service):
        """Start a single service"""
        service_path = project_root / service['path']
        
        print(f"Starting {service['name']} on port {service['port']}...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(service['port'])
            env['SERVICE_NAME'] = service['name']
            
            process = subprocess.Popen(
                service['cmd'],
                cwd=service_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append({
                'name': service['name'],
                'process': process,
                'port': service['port']
            })
            
            print(f"✓ {service['name']} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start {service['name']}: {e}")
            return False
    
    def start_all_services(self):
        """Start all services"""
        print("Starting all Lift OS Core services...")
        
        for service in self.services:
            if not self.start_service(service):
                return False
            time.sleep(2)  # Wait between service starts
        
        print("\n" + "="*50)
        print("All services started successfully!")
        print("="*50)
        
        for proc_info in self.processes:
            print(f"{proc_info['name']}: http://localhost:{proc_info['port']}")
        
        print("\nPress Ctrl+C to stop all services")
        return True
    
    def stop_all_services(self):
        """Stop all services"""
        print("\nStopping all services...")
        
        for proc_info in self.processes:
            try:
                proc_info['process'].terminate()
                proc_info['process'].wait(timeout=5)
                print(f"✓ Stopped {proc_info['name']}")
            except subprocess.TimeoutExpired:
                proc_info['process'].kill()
                print(f"✓ Force stopped {proc_info['name']}")
            except Exception as e:
                print(f"✗ Error stopping {proc_info['name']}: {e}")
    
    def monitor_services(self):
        """Monitor service health"""
        while True:
            try:
                time.sleep(10)
                for proc_info in self.processes:
                    if proc_info['process'].poll() is not None:
                        print(f"⚠ Service {proc_info['name']} has stopped unexpectedly")
            except KeyboardInterrupt:
                break

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    sys.exit(0)

def main():
    """Main function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    manager = ServiceManager()
    
    try:
        # Install dependencies
        if not manager.install_dependencies():
            print("Failed to install dependencies. Exiting.")
            return 1
        
        # Start all services
        if not manager.start_all_services():
            print("Failed to start all services. Exiting.")
            return 1
        
        # Monitor services
        manager.monitor_services()
        
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all_services()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())