#!/usr/bin/env python3
"""
Deploy Phase 3 & Phase 4 Services
Comprehensive deployment script for LiftOS Phase 3 (True Intelligence) and Phase 4 (Complete Observability)
"""

import subprocess
import time
import sys
import os
from pathlib import Path

class LiftOSDeployer:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.services_dir = self.base_dir / "services"
        
        # Phase 3 services (existing enhanced services)
        self.phase3_services = [
            "memory",
            "gateway",
            "intelligence"
        ]
        
        # Phase 4 services (new transparency services)
        self.phase4_services = [
            "business-metrics",      # Audit Service
            "user-analytics",        # Explanation Engine  
            "impact-monitoring",     # Monitoring Service
            "strategic-intelligence", # Validation Service
            "business-intelligence"  # Transparency Dashboard
        ]
        
        self.all_services = self.phase3_services + self.phase4_services
        
    def run_command(self, command, cwd=None, check=True):
        """Run a command and return the result"""
        try:
            print(f"Running: {command}")
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.base_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            if result.stdout:
                print(f"Output: {result.stdout}")
            if result.stderr and result.returncode != 0:
                print(f"Error: {result.stderr}")
                
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)
                
            return result
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            return None
        except Exception as e:
            print(f"Command failed: {command} - {str(e)}")
            if check:
                raise
            return None
    
    def check_docker_status(self):
        """Check if Docker is running"""
        print("Checking Docker status...")
        try:
            result = self.run_command("docker info", check=False)
            if result and result.returncode == 0:
                print("[OK] Docker is running")
                return True
            else:
                print("[FAIL] Docker is not running or not accessible")
                return False
        except Exception:
            print("[FAIL] Docker is not available")
            return False
    
    def build_service(self, service_name):
        """Build a Docker image for a service"""
        service_dir = self.services_dir / service_name
        if not service_dir.exists():
            print(f"[FAIL] Service directory not found: {service_dir}")
            return False
            
        dockerfile_path = service_dir / "Dockerfile"
        if not dockerfile_path.exists():
            print(f"[FAIL] Dockerfile not found for service: {service_name}")
            return False
            
        print(f"Building {service_name} service...")
        image_name = f"liftos/{service_name}:latest"
        
        try:
            result = self.run_command(
                f"docker build -t {image_name} -f services/{service_name}/Dockerfile .",
                cwd=self.base_dir
            )
            if result and result.returncode == 0:
                print(f"[OK] Built {service_name} successfully")
                return True
            else:
                print(f"[FAIL] Failed to build {service_name}")
                return False
        except Exception as e:
            print(f"[FAIL] Error building {service_name}: {str(e)}")
            return False
    
    def create_docker_network(self):
        """Create Docker network for services"""
        print("Creating Docker network...")
        try:
            # Check if network exists
            result = self.run_command("docker network ls --filter name=liftos-network", check=False)
            if result and "liftos-network" in result.stdout:
                print("[OK] Network liftos-network already exists")
                return True
                
            # Create network
            result = self.run_command("docker network create liftos-network")
            if result and result.returncode == 0:
                print("[OK] Created liftos-network")
                return True
            else:
                print("[FAIL] Failed to create network")
                return False
        except Exception as e:
            print(f"[FAIL] Error creating network: {str(e)}")
            return False
    
    def deploy_service(self, service_name, port):
        """Deploy a service container"""
        print(f"Deploying {service_name} on port {port}...")
        
        # Stop existing container if running
        self.run_command(f"docker stop liftos-{service_name}", check=False)
        self.run_command(f"docker rm liftos-{service_name}", check=False)
        
        # Run new container
        image_name = f"liftos/{service_name}:latest"
        container_name = f"liftos-{service_name}"
        
        docker_run_cmd = f"""docker run -d \
            --name {container_name} \
            --network liftos-network \
            -p {port}:{port} \
            -e PORT={port} \
            {image_name}"""
        
        try:
            result = self.run_command(docker_run_cmd)
            if result and result.returncode == 0:
                print(f"[OK] Deployed {service_name} on port {port}")
                return True
            else:
                print(f"[FAIL] Failed to deploy {service_name}")
                return False
        except Exception as e:
            print(f"[FAIL] Error deploying {service_name}: {str(e)}")
            return False
    
    def wait_for_service(self, service_name, port, timeout=60):
        """Wait for a service to be healthy"""
        print(f"Waiting for {service_name} to be ready...")
        
        import requests
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[OK] {service_name} is healthy")
                    return True
            except:
                pass
            time.sleep(2)
        
        print(f"[FAIL] {service_name} failed to become healthy within {timeout}s")
        return False
    
    def deploy_phase3(self):
        """Deploy Phase 3: True Intelligence services"""
        print("\n" + "="*60)
        print("DEPLOYING PHASE 3: TRUE INTELLIGENCE")
        print("="*60)
        
        phase3_ports = {
            "memory": 8003,
            "gateway": 8000,
            "intelligence": 8010
        }
        
        success_count = 0
        for service in self.phase3_services:
            if service in phase3_ports:
                if self.build_service(service):
                    if self.deploy_service(service, phase3_ports[service]):
                        if self.wait_for_service(service, phase3_ports[service]):
                            success_count += 1
        
        print(f"\nPhase 3 Deployment: {success_count}/{len(self.phase3_services)} services successful")
        return success_count == len(self.phase3_services)
    
    def deploy_phase4(self):
        """Deploy Phase 4: Complete Observability services"""
        print("\n" + "="*60)
        print("DEPLOYING PHASE 4: COMPLETE OBSERVABILITY")
        print("="*60)
        
        phase4_ports = {
            "business-metrics": 8012,      # Audit Service
            "user-analytics": 8013,        # Explanation Engine
            "impact-monitoring": 8014,     # Monitoring Service
            "strategic-intelligence": 8015, # Validation Service
            "business-intelligence": 8016  # Transparency Dashboard
        }
        
        success_count = 0
        for service in self.phase4_services:
            if service in phase4_ports:
                if self.build_service(service):
                    if self.deploy_service(service, phase4_ports[service]):
                        if self.wait_for_service(service, phase4_ports[service]):
                            success_count += 1
        
        print(f"\nPhase 4 Deployment: {success_count}/{len(self.phase4_services)} services successful")
        return success_count == len(self.phase4_services)
    
    def validate_deployment(self):
        """Validate the complete deployment"""
        print("\n" + "="*60)
        print("VALIDATING DEPLOYMENT")
        print("="*60)
        
        # Run the existing validation script
        try:
            result = self.run_command("python validate_next_phase_complete.py")
            if result and result.returncode == 0:
                print("[OK] Deployment validation successful")
                return True
            else:
                print("[FAIL] Deployment validation failed")
                return False
        except Exception as e:
            print(f"[FAIL] Validation error: {str(e)}")
            return False
    
    def deploy_all(self):
        """Deploy both Phase 3 and Phase 4"""
        print("="*60)
        print("LIFTOS PHASE 3 & PHASE 4 DEPLOYMENT")
        print("="*60)
        
        # Check prerequisites
        if not self.check_docker_status():
            print("[FAIL] Docker is required but not running. Please start Docker Desktop.")
            return False
        
        if not self.create_docker_network():
            print("[FAIL] Failed to create Docker network")
            return False
        
        # Deploy phases
        phase3_success = self.deploy_phase3()
        phase4_success = self.deploy_phase4()
        
        # Validate deployment
        if phase3_success and phase4_success:
            validation_success = self.validate_deployment()
            
            print("\n" + "="*60)
            print("DEPLOYMENT SUMMARY")
            print("="*60)
            print(f"Phase 3 (True Intelligence): {'[OK] SUCCESS' if phase3_success else '[FAIL] FAILED'}")
            print(f"Phase 4 (Complete Observability): {'[OK] SUCCESS' if phase4_success else '[FAIL] FAILED'}")
            print(f"Validation: {'[OK] SUCCESS' if validation_success else '[FAIL] FAILED'}")
            
            if phase3_success and phase4_success and validation_success:
                print("\n[SUCCESS] DEPLOYMENT COMPLETE - Both phases successfully deployed!")
                return True
            else:
                print("\n[WARNING] DEPLOYMENT PARTIAL - Some issues detected")
                return False
        else:
            print("\n[FAIL] DEPLOYMENT FAILED - Critical services failed to deploy")
            return False

def main():
    deployer = LiftOSDeployer()
    success = deployer.deploy_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()