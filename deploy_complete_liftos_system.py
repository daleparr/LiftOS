#!/usr/bin/env python3
"""
LiftOS Complete System Deployment Script

This script orchestrates the deployment of the complete enhanced LiftOS system,
integrating all phases (Phase 1-3) into a unified, production-ready platform.

Usage:
    python deploy_complete_liftos_system.py --environment production
    python deploy_complete_liftos_system.py --environment staging --dry-run
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml
import requests
from pathlib import Path

class LiftOSDeploymentOrchestrator:
    """Orchestrates the complete LiftOS system deployment"""
    
    def __init__(self, environment: str = "production", dry_run: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.deployment_start_time = datetime.now()
        self.deployment_log = []
        
        # Deployment configuration
        self.config = {
            "production": {
                "namespace": "liftos-production",
                "replicas": 3,
                "resources": {
                    "requests": {"memory": "512Mi", "cpu": "250m"},
                    "limits": {"memory": "1Gi", "cpu": "500m"}
                },
                "domain": "app.liftos.com",
                "api_domain": "api.liftos.com"
            },
            "staging": {
                "namespace": "liftos-staging",
                "replicas": 2,
                "resources": {
                    "requests": {"memory": "256Mi", "cpu": "125m"},
                    "limits": {"memory": "512Mi", "cpu": "250m"}
                },
                "domain": "staging.liftos.com",
                "api_domain": "api-staging.liftos.com"
            }
        }
        
        self.current_config = self.config[environment]
        
    def log_step(self, step: str, status: str = "INFO", details: str = ""):
        """Log deployment step with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "details": details
        }
        self.deployment_log.append(log_entry)
        
        status_emoji = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "PROGRESS": "🔄"
        }
        
        print(f"{status_emoji.get(status, 'ℹ️')} [{timestamp}] {step}")
        if details:
            print(f"   {details}")
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging"""
        self.log_step(f"Executing: {command}", "PROGRESS")
        
        if self.dry_run:
            self.log_step("DRY RUN: Command not executed", "INFO")
            return subprocess.CompletedProcess(command, 0, "", "")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.returncode == 0:
                self.log_step("Command executed successfully", "SUCCESS")
            else:
                self.log_step(f"Command failed with code {result.returncode}", "ERROR", result.stderr)
            
            return result
        except subprocess.CalledProcessError as e:
            self.log_step(f"Command failed: {e}", "ERROR")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        self.log_step("Checking deployment prerequisites", "PROGRESS")
        
        prerequisites = [
            ("kubectl", "kubectl version --client"),
            ("docker", "docker --version"),
            ("helm", "helm version"),
        ]
        
        all_good = True
        for tool, command in prerequisites:
            try:
                result = self.run_command(command, check=False)
                if result.returncode == 0:
                    self.log_step(f"{tool} is available", "SUCCESS")
                else:
                    self.log_step(f"{tool} is not available", "ERROR")
                    all_good = False
            except Exception as e:
                self.log_step(f"Error checking {tool}: {e}", "ERROR")
                all_good = False
        
        # Check Kubernetes cluster connectivity
        try:
            result = self.run_command("kubectl cluster-info", check=False)
            if result.returncode == 0:
                self.log_step("Kubernetes cluster is accessible", "SUCCESS")
            else:
                self.log_step("Cannot connect to Kubernetes cluster", "ERROR")
                all_good = False
        except Exception as e:
            self.log_step(f"Kubernetes connectivity error: {e}", "ERROR")
            all_good = False
        
        return all_good
    
    def create_namespace(self) -> bool:
        """Create Kubernetes namespace"""
        self.log_step(f"Creating namespace: {self.current_config['namespace']}", "PROGRESS")
        
        try:
            # Check if namespace exists
            result = self.run_command(
                f"kubectl get namespace {self.current_config['namespace']}", 
                check=False
            )
            
            if result.returncode == 0:
                self.log_step("Namespace already exists", "INFO")
                return True
            
            # Create namespace
            self.run_command(f"kubectl create namespace {self.current_config['namespace']}")
            self.log_step("Namespace created successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to create namespace: {e}", "ERROR")
            return False
    
    def deploy_infrastructure(self) -> bool:
        """Deploy infrastructure components"""
        self.log_step("Deploying infrastructure components", "PROGRESS")
        
        infrastructure_components = [
            "postgresql",
            "redis",
            "nginx-ingress",
            "cert-manager"
        ]
        
        try:
            for component in infrastructure_components:
                self.log_step(f"Deploying {component}", "PROGRESS")
                
                if component == "postgresql":
                    self.run_command(
                        f"helm upgrade --install postgresql bitnami/postgresql "
                        f"--namespace {self.current_config['namespace']} "
                        f"--set auth.postgresPassword=liftos-secure-password "
                        f"--set primary.persistence.size=20Gi"
                    )
                
                elif component == "redis":
                    self.run_command(
                        f"helm upgrade --install redis bitnami/redis "
                        f"--namespace {self.current_config['namespace']} "
                        f"--set auth.password=liftos-redis-password "
                        f"--set master.persistence.size=8Gi"
                    )
                
                elif component == "nginx-ingress":
                    self.run_command(
                        f"helm upgrade --install nginx-ingress ingress-nginx/ingress-nginx "
                        f"--namespace {self.current_config['namespace']}"
                    )
                
                elif component == "cert-manager":
                    self.run_command(
                        f"helm upgrade --install cert-manager jetstack/cert-manager "
                        f"--namespace cert-manager --create-namespace "
                        f"--set installCRDs=true"
                    )
                
                self.log_step(f"{component} deployed successfully", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log_step(f"Infrastructure deployment failed: {e}", "ERROR")
            return False
    
    def build_and_push_images(self) -> bool:
        """Build and push Docker images"""
        self.log_step("Building and pushing Docker images", "PROGRESS")
        
        images = [
            {
                "name": "liftos-streamlit",
                "path": "liftos-streamlit",
                "dockerfile": "liftos-streamlit/Dockerfile"
            },
            {
                "name": "liftos-api",
                "path": "services",
                "dockerfile": "services/Dockerfile"
            }
        ]
        
        try:
            for image in images:
                self.log_step(f"Building {image['name']}", "PROGRESS")
                
                # Build image
                self.run_command(
                    f"docker build -t liftos/{image['name']}:latest "
                    f"-f {image['dockerfile']} {image['path']}"
                )
                
                # Tag for registry
                self.run_command(
                    f"docker tag liftos/{image['name']}:latest "
                    f"registry.liftos.com/{image['name']}:latest"
                )
                
                # Push to registry
                self.run_command(
                    f"docker push registry.liftos.com/{image['name']}:latest"
                )
                
                self.log_step(f"{image['name']} built and pushed successfully", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log_step(f"Image build/push failed: {e}", "ERROR")
            return False
    
    def deploy_backend_services(self) -> bool:
        """Deploy backend microservices"""
        self.log_step("Deploying backend microservices", "PROGRESS")
        
        services = [
            "intelligence",
            "memory", 
            "data-ingestion",
            "observability",
            "business-intelligence",
            "gateway"
        ]
        
        try:
            for service in services:
                self.log_step(f"Deploying {service} service", "PROGRESS")
                
                # Apply Kubernetes manifests
                self.run_command(
                    f"kubectl apply -f k8s/services/{service}/ "
                    f"--namespace {self.current_config['namespace']}"
                )
                
                # Wait for deployment to be ready
                self.run_command(
                    f"kubectl wait --for=condition=available deployment/{service} "
                    f"--namespace {self.current_config['namespace']} "
                    f"--timeout=300s"
                )
                
                self.log_step(f"{service} service deployed successfully", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log_step(f"Backend services deployment failed: {e}", "ERROR")
            return False
    
    def deploy_frontend_application(self) -> bool:
        """Deploy Streamlit frontend application"""
        self.log_step("Deploying Streamlit frontend application", "PROGRESS")
        
        try:
            # Create Streamlit deployment manifest
            streamlit_manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "liftos-streamlit",
                    "namespace": self.current_config["namespace"]
                },
                "spec": {
                    "replicas": self.current_config["replicas"],
                    "selector": {
                        "matchLabels": {
                            "app": "liftos-streamlit"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "liftos-streamlit"
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": "streamlit",
                                "image": "registry.liftos.com/liftos-streamlit:latest",
                                "ports": [{"containerPort": 8501}],
                                "env": [
                                    {"name": "LIFTOS_ENV", "value": self.environment},
                                    {"name": "LIFTOS_API_BASE_URL", "value": f"https://{self.current_config['api_domain']}"}
                                ],
                                "resources": self.current_config["resources"]
                            }]
                        }
                    }
                }
            }
            
            # Write manifest to file
            with open("streamlit-deployment.yaml", "w") as f:
                yaml.dump(streamlit_manifest, f)
            
            # Apply deployment
            self.run_command(
                f"kubectl apply -f streamlit-deployment.yaml "
                f"--namespace {self.current_config['namespace']}"
            )
            
            # Create service
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "liftos-streamlit-service",
                    "namespace": self.current_config["namespace"]
                },
                "spec": {
                    "selector": {
                        "app": "liftos-streamlit"
                    },
                    "ports": [{
                        "port": 80,
                        "targetPort": 8501
                    }],
                    "type": "ClusterIP"
                }
            }
            
            with open("streamlit-service.yaml", "w") as f:
                yaml.dump(service_manifest, f)
            
            self.run_command(
                f"kubectl apply -f streamlit-service.yaml "
                f"--namespace {self.current_config['namespace']}"
            )
            
            # Wait for deployment
            self.run_command(
                f"kubectl wait --for=condition=available deployment/liftos-streamlit "
                f"--namespace {self.current_config['namespace']} "
                f"--timeout=300s"
            )
            
            self.log_step("Frontend application deployed successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Frontend deployment failed: {e}", "ERROR")
            return False
    
    def configure_ingress(self) -> bool:
        """Configure ingress for external access"""
        self.log_step("Configuring ingress for external access", "PROGRESS")
        
        try:
            ingress_manifest = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": "liftos-ingress",
                    "namespace": self.current_config["namespace"],
                    "annotations": {
                        "kubernetes.io/ingress.class": "nginx",
                        "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                        "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                    }
                },
                "spec": {
                    "tls": [{
                        "hosts": [self.current_config["domain"]],
                        "secretName": "liftos-tls"
                    }],
                    "rules": [{
                        "host": self.current_config["domain"],
                        "http": {
                            "paths": [{
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": "liftos-streamlit-service",
                                        "port": {"number": 80}
                                    }
                                }
                            }]
                        }
                    }]
                }
            }
            
            with open("liftos-ingress.yaml", "w") as f:
                yaml.dump(ingress_manifest, f)
            
            self.run_command(
                f"kubectl apply -f liftos-ingress.yaml "
                f"--namespace {self.current_config['namespace']}"
            )
            
            self.log_step("Ingress configured successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Ingress configuration failed: {e}", "ERROR")
            return False
    
    def run_database_migrations(self) -> bool:
        """Run database migrations"""
        self.log_step("Running database migrations", "PROGRESS")
        
        try:
            # Run migrations for each service
            services_with_migrations = ["intelligence", "memory", "data-ingestion"]
            
            for service in services_with_migrations:
                self.log_step(f"Running migrations for {service}", "PROGRESS")
                
                self.run_command(
                    f"kubectl exec -it deployment/{service} "
                    f"--namespace {self.current_config['namespace']} "
                    f"-- python manage.py migrate"
                )
                
                self.log_step(f"Migrations completed for {service}", "SUCCESS")
            
            # Initialize KSE memory system
            self.log_step("Initializing KSE memory system", "PROGRESS")
            self.run_command(
                f"kubectl exec -it deployment/memory "
                f"--namespace {self.current_config['namespace']} "
                f"-- python init_kse.py"
            )
            
            self.log_step("Database migrations completed successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Database migrations failed: {e}", "ERROR")
            return False
    
    def validate_deployment(self) -> bool:
        """Validate deployment health and functionality"""
        self.log_step("Validating deployment health", "PROGRESS")
        
        try:
            # Check pod status
            result = self.run_command(
                f"kubectl get pods --namespace {self.current_config['namespace']}", 
                check=False
            )
            
            if "Running" in result.stdout:
                self.log_step("Pods are running", "SUCCESS")
            else:
                self.log_step("Some pods are not running", "WARNING")
            
            # Check service endpoints
            self.run_command(
                f"kubectl get endpoints --namespace {self.current_config['namespace']}"
            )
            
            # Test application health
            if not self.dry_run:
                time.sleep(30)  # Wait for services to be ready
                
                try:
                    response = requests.get(f"https://{self.current_config['domain']}/health", timeout=30)
                    if response.status_code == 200:
                        self.log_step("Application health check passed", "SUCCESS")
                    else:
                        self.log_step(f"Application health check failed: {response.status_code}", "WARNING")
                except requests.RequestException as e:
                    self.log_step(f"Application health check error: {e}", "WARNING")
            
            self.log_step("Deployment validation completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Deployment validation failed: {e}", "ERROR")
            return False
    
    def generate_deployment_report(self) -> str:
        """Generate deployment report"""
        deployment_duration = datetime.now() - self.deployment_start_time
        
        report = f"""
# LiftOS Complete System Deployment Report

**Deployment Date:** {self.deployment_start_time.strftime('%Y-%m-%d %H:%M:%S')}
**Environment:** {self.environment}
**Duration:** {deployment_duration}
**Status:** {'DRY RUN' if self.dry_run else 'COMPLETED'}

## Deployment Summary

- **Namespace:** {self.current_config['namespace']}
- **Replicas:** {self.current_config['replicas']}
- **Domain:** {self.current_config['domain']}
- **API Domain:** {self.current_config['api_domain']}

## Deployment Steps

"""
        
        for log_entry in self.deployment_log:
            status_icon = {
                "SUCCESS": "✅",
                "ERROR": "❌", 
                "WARNING": "⚠️",
                "INFO": "ℹ️",
                "PROGRESS": "🔄"
            }.get(log_entry["status"], "ℹ️")
            
            report += f"- {status_icon} **{log_entry['step']}** ({log_entry['timestamp']})\n"
            if log_entry["details"]:
                report += f"  - {log_entry['details']}\n"
        
        report += f"""

## Access Information

- **Frontend URL:** https://{self.current_config['domain']}
- **API URL:** https://{self.current_config['api_domain']}
- **Kubernetes Namespace:** {self.current_config['namespace']}

## Next Steps

1. Verify all dashboards are accessible
2. Test optimization workflows
3. Validate platform integrations
4. Monitor system performance
5. Conduct user training sessions

---
*Report generated by LiftOS Deployment Orchestrator*
"""
        
        return report
    
    def deploy(self) -> bool:
        """Execute complete deployment"""
        self.log_step("Starting LiftOS complete system deployment", "INFO")
        
        deployment_steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Create Namespace", self.create_namespace),
            ("Deploy Infrastructure", self.deploy_infrastructure),
            ("Build and Push Images", self.build_and_push_images),
            ("Deploy Backend Services", self.deploy_backend_services),
            ("Deploy Frontend Application", self.deploy_frontend_application),
            ("Configure Ingress", self.configure_ingress),
            ("Run Database Migrations", self.run_database_migrations),
            ("Validate Deployment", self.validate_deployment)
        ]
        
        for step_name, step_function in deployment_steps:
            self.log_step(f"Starting: {step_name}", "PROGRESS")
            
            try:
                success = step_function()
                if success:
                    self.log_step(f"Completed: {step_name}", "SUCCESS")
                else:
                    self.log_step(f"Failed: {step_name}", "ERROR")
                    return False
            except Exception as e:
                self.log_step(f"Error in {step_name}: {e}", "ERROR")
                return False
        
        # Generate and save deployment report
        report = self.generate_deployment_report()
        report_filename = f"liftos_deployment_report_{self.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_filename, "w") as f:
            f.write(report)
        
        self.log_step(f"Deployment report saved: {report_filename}", "INFO")
        self.log_step("LiftOS complete system deployment finished successfully!", "SUCCESS")
        
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy LiftOS Complete System")
    parser.add_argument(
        "--environment", 
        choices=["production", "staging"], 
        default="production",
        help="Deployment environment"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform dry run without executing commands"
    )
    
    args = parser.parse_args()
    
    print("🚀 LiftOS Complete System Deployment Orchestrator")
    print("=" * 50)
    print(f"Environment: {args.environment}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 50)
    
    # Create deployment orchestrator
    orchestrator = LiftOSDeploymentOrchestrator(
        environment=args.environment,
        dry_run=args.dry_run
    )
    
    # Execute deployment
    try:
        success = orchestrator.deploy()
        
        if success:
            print("\n🎉 Deployment completed successfully!")
            print(f"🌐 Access your LiftOS system at: https://{orchestrator.current_config['domain']}")
        else:
            print("\n❌ Deployment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()