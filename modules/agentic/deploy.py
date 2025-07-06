#!/usr/bin/env python3
"""
Deployment script for the LiftOS Agentic microservice.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {command}")
    if capture_output:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result.stdout.strip()
    else:
        result = subprocess.run(command, shell=True)
        if check and result.returncode != 0:
            sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check Docker
    try:
        run_command("docker --version", capture_output=True)
        print("✓ Docker is installed")
    except:
        print("✗ Docker is not installed or not in PATH")
        return False
    
    # Check Docker Compose
    try:
        run_command("docker-compose --version", capture_output=True)
        print("✓ Docker Compose is installed")
    except:
        print("✗ Docker Compose is not installed or not in PATH")
        return False
    
    return True


def setup_environment():
    """Set up environment configuration."""
    print("Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("Creating .env file from .env.example...")
            run_command(f"cp {env_example} {env_file}")
            print("✓ .env file created")
            print("⚠️  Please edit .env file with your configuration before proceeding")
            return False
        else:
            print("✗ .env.example file not found")
            return False
    else:
        print("✓ .env file already exists")
    
    return True


def build_images():
    """Build Docker images."""
    print("Building Docker images...")
    run_command("docker-compose build")
    print("✓ Docker images built successfully")


def start_services(detached=True):
    """Start the services."""
    print("Starting services...")
    
    if detached:
        run_command("docker-compose up -d")
    else:
        run_command("docker-compose up")
    
    print("✓ Services started")


def stop_services():
    """Stop the services."""
    print("Stopping services...")
    run_command("docker-compose down")
    print("✓ Services stopped")


def check_health():
    """Check service health."""
    print("Checking service health...")
    
    max_retries = 30
    retry_interval = 2
    
    for i in range(max_retries):
        try:
            result = run_command("curl -f http://localhost:8007/health", capture_output=True, check=False)
            if "healthy" in result.lower():
                print("✓ Agentic service is healthy")
                return True
        except:
            pass
        
        print(f"Waiting for service to be ready... ({i+1}/{max_retries})")
        time.sleep(retry_interval)
    
    print("✗ Service health check failed")
    return False


def show_logs():
    """Show service logs."""
    print("Showing service logs...")
    run_command("docker-compose logs -f agentic")


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Install test dependencies
    run_command("pip install pytest pytest-asyncio pytest-mock pytest-cov")
    
    # Run tests
    run_command("pytest tests/ -v --cov=. --cov-report=term-missing")
    
    print("✓ Tests completed")


def clean_up():
    """Clean up Docker resources."""
    print("Cleaning up...")
    
    # Stop and remove containers
    run_command("docker-compose down -v", check=False)
    
    # Remove images
    run_command("docker rmi $(docker images -q liftos-agentic*) 2>/dev/null || true", check=False)
    
    # Remove volumes
    run_command("docker volume prune -f", check=False)
    
    print("✓ Cleanup completed")


def show_status():
    """Show service status."""
    print("Service Status:")
    print("=" * 50)
    
    # Show running containers
    run_command("docker-compose ps")
    
    print("\nService URLs:")
    print("- Agentic API: http://localhost:8007")
    print("- Health Check: http://localhost:8007/health")
    print("- API Docs: http://localhost:8007/docs")
    print("- Metrics: http://localhost:8007/metrics")
    
    print("\nDatabase:")
    print("- PostgreSQL: localhost:5433")
    print("- Redis: localhost:6380")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy LiftOS Agentic microservice")
    parser.add_argument("action", choices=[
        "setup", "build", "start", "stop", "restart", 
        "logs", "test", "clean", "status", "health"
    ], help="Action to perform")
    parser.add_argument("--no-detach", action="store_true", help="Don't run in detached mode")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.action == "setup":
        if not check_dependencies():
            sys.exit(1)
        if not setup_environment():
            print("\nPlease configure your .env file and run 'python deploy.py build' to continue")
            sys.exit(1)
        print("\n✓ Setup completed successfully")
        
    elif args.action == "build":
        if not check_dependencies():
            sys.exit(1)
        build_images()
        
    elif args.action == "start":
        if not check_dependencies():
            sys.exit(1)
        start_services(detached=not args.no_detach)
        if not args.no_detach:
            time.sleep(5)
            check_health()
        
    elif args.action == "stop":
        stop_services()
        
    elif args.action == "restart":
        stop_services()
        start_services(detached=not args.no_detach)
        if not args.no_detach:
            time.sleep(5)
            check_health()
        
    elif args.action == "logs":
        show_logs()
        
    elif args.action == "test":
        run_tests()
        
    elif args.action == "clean":
        clean_up()
        
    elif args.action == "status":
        show_status()
        
    elif args.action == "health":
        if check_health():
            print("✓ Service is healthy")
        else:
            print("✗ Service is not healthy")
            sys.exit(1)


if __name__ == "__main__":
    main()