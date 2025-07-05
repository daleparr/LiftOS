"""
Start multiple services for Priority 2 testing
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Manage multiple services for testing"""
    
    def __init__(self):
        self.processes = {}
        self.services = {
            "gateway": {"port": 8000, "path": "services/gateway"},
            "auth": {"port": 8001, "path": "services/auth"},
            "memory": {"port": 8002, "path": "services/memory"},
            "registry": {"port": 8004, "path": "services/registry"}
        }
    
    def start_service(self, service_name: str):
        """Start a single service"""
        if service_name in self.processes:
            logger.warning(f"Service {service_name} is already running")
            return
        
        service_config = self.services.get(service_name)
        if not service_config:
            logger.error(f"Unknown service: {service_name}")
            return
        
        service_path = project_root / service_config["path"]
        port = service_config["port"]
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        # Start the service
        cmd = [
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload"
        ]
        
        logger.info(f"Starting {service_name} on port {port}...")
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=service_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = {
                "process": process,
                "port": port,
                "path": service_path
            }
            
            logger.info(f"✓ {service_name} started (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
    
    def stop_service(self, service_name: str):
        """Stop a single service"""
        if service_name not in self.processes:
            logger.warning(f"Service {service_name} is not running")
            return
        
        process_info = self.processes[service_name]
        process = process_info["process"]
        
        logger.info(f"Stopping {service_name}...")
        
        try:
            process.terminate()
            process.wait(timeout=10)
            logger.info(f"✓ {service_name} stopped")
        except subprocess.TimeoutExpired:
            logger.warning(f"Force killing {service_name}...")
            process.kill()
            process.wait()
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
        
        del self.processes[service_name]
    
    def start_all(self):
        """Start all services"""
        logger.info("Starting all services for Priority 2 testing...")
        
        # Start services in order
        for service_name in ["gateway", "auth", "memory", "registry"]:
            self.start_service(service_name)
            time.sleep(2)  # Give each service time to start
        
        logger.info("\n" + "="*50)
        logger.info("PRIORITY 2 SERVICES STARTED")
        logger.info("="*50)
        
        for service_name, process_info in self.processes.items():
            port = process_info["port"]
            logger.info(f"• {service_name}: http://localhost:{port}")
        
        logger.info("\nServices ready for Priority 2 testing!")
        logger.info("Press Ctrl+C to stop all services")
    
    def stop_all(self):
        """Stop all services"""
        logger.info("Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        logger.info("All services stopped")
    
    def status(self):
        """Show status of all services"""
        if not self.processes:
            logger.info("No services running")
            return
        
        logger.info("Running services:")
        for service_name, process_info in self.processes.items():
            process = process_info["process"]
            port = process_info["port"]
            
            if process.poll() is None:
                status = "RUNNING"
            else:
                status = "STOPPED"
            
            logger.info(f"• {service_name}: {status} (port {port})")

def main():
    """Main function"""
    manager = ServiceManager()
    
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "start":
                if len(sys.argv) > 2:
                    service_name = sys.argv[2]
                    manager.start_service(service_name)
                else:
                    manager.start_all()
            
            elif command == "stop":
                if len(sys.argv) > 2:
                    service_name = sys.argv[2]
                    manager.stop_service(service_name)
                else:
                    manager.stop_all()
            
            elif command == "status":
                manager.status()
            
            elif command == "restart":
                manager.stop_all()
                time.sleep(2)
                manager.start_all()
            
            else:
                print("Usage: python start_priority2_services.py [start|stop|status|restart] [service_name]")
                print("Services: gateway, auth, memory, registry")
        
        else:
            # Default: start all and wait
            manager.start_all()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nReceived interrupt signal")
    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    
    finally:
        manager.stop_all()

if __name__ == "__main__":
    main()