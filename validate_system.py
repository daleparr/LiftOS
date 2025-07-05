#!/usr/bin/env python3
"""
Lift OS Core - System Validation Script

This script performs basic validation of the Lift OS Core system without requiring
external services like databases or Redis to be running.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
import json

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print a success message"""
    print(f"[OK] {message}")

def print_error(message: str):
    """Print an error message"""
    print(f"[ERROR] {message}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"[WARNING] {message}")

def print_info(message: str):
    """Print an info message"""
    print(f"[INFO] {message}")

def validate_file_structure() -> bool:
    """Validate the project file structure"""
    print_header("File Structure Validation")
    
    required_files = [
        "README.md",
        "docker-compose.yml",
        "docker-compose.prod.yml",
        ".env.example",
        "Makefile",
        "requirements.txt"
    ]
    
    required_dirs = [
        "services",
        "ui-shell",
        "modules",
        "tests",
        "docs",
        "k8s",
        "monitoring",
        "shared"
    ]
    
    all_valid = True
    
    # Check required files
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"Found {file_path}")
        else:
            print_error(f"Missing {file_path}")
            all_valid = False
    
    # Check required directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print_success(f"Found directory {dir_path}/")
        else:
            print_error(f"Missing directory {dir_path}/")
            all_valid = False
    
    return all_valid

def validate_services() -> bool:
    """Validate service structure and basic imports"""
    print_header("Services Validation")
    
    services = [
        "gateway",
        "auth", 
        "memory",
        "registry",
        "billing",
        "observability"
    ]
    
    all_valid = True
    
    for service in services:
        service_dir = f"services/{service}"
        if not os.path.isdir(service_dir):
            print_error(f"Missing service directory: {service_dir}")
            all_valid = False
            continue
        
        # Check required files
        required_files = ["app.py", "requirements.txt", "Dockerfile"]
        service_valid = True
        
        for file_name in required_files:
            file_path = f"{service_dir}/{file_name}"
            if os.path.exists(file_path):
                print_success(f"Found {service}/{file_name}")
            else:
                print_error(f"Missing {service}/{file_name}")
                service_valid = False
        
        # Try to validate Python syntax
        app_file = f"{service_dir}/app.py"
        if os.path.exists(app_file):
            try:
                with open(app_file, 'r') as f:
                    content = f.read()
                    compile(content, app_file, 'exec')
                print_success(f"Valid Python syntax in {service}/app.py")
            except SyntaxError as e:
                print_error(f"Syntax error in {service}/app.py: {e}")
                service_valid = False
            except Exception as e:
                print_warning(f"Could not validate {service}/app.py: {e}")
        
        if not service_valid:
            all_valid = False
    
    return all_valid

def validate_modules() -> bool:
    """Validate module structure"""
    print_header("Modules Validation")
    
    modules_dir = "modules"
    if not os.path.isdir(modules_dir):
        print_error(f"Missing modules directory")
        return False
    
    modules = []
    for item in os.listdir(modules_dir):
        item_path = os.path.join(modules_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            modules.append(item)
    
    if not modules:
        print_warning("No modules found")
        return True
    
    all_valid = True
    
    for module in modules:
        module_dir = f"modules/{module}"
        print_info(f"Validating module: {module}")
        
        # Check required files
        required_files = ["app.py", "module.json", "requirements.txt", "Dockerfile"]
        module_valid = True
        
        for file_name in required_files:
            file_path = f"{module_dir}/{file_name}"
            if os.path.exists(file_path):
                print_success(f"  Found {file_name}")
            else:
                print_error(f"  Missing {file_name}")
                module_valid = False
        
        # Validate module.json
        module_json_path = f"{module_dir}/module.json"
        if os.path.exists(module_json_path):
            try:
                with open(module_json_path, 'r') as f:
                    module_config = json.load(f)
                
                # Check required top-level keys
                required_keys = ["name", "version"]
                for key in required_keys:
                    if key in module_config:
                        print_success(f"  Valid {key} in module.json")
                    else:
                        print_error(f"  Missing {key} in module.json")
                        module_valid = False
                
                # Check for description in metadata
                if "metadata" in module_config and "description" in module_config["metadata"]:
                    print_success(f"  Valid description in module.json")
                elif "description" in module_config:
                    print_success(f"  Valid description in module.json")
                else:
                    print_error(f"  Missing description in module.json")
                    module_valid = False
                        
            except json.JSONDecodeError as e:
                print_error(f"  Invalid JSON in module.json: {e}")
                module_valid = False
        
        if not module_valid:
            all_valid = False
    
    return all_valid

def validate_ui_shell() -> bool:
    """Validate UI shell structure"""
    print_header("UI Shell Validation")
    
    ui_dir = "ui-shell"
    if not os.path.isdir(ui_dir):
        print_error("Missing ui-shell directory")
        return False
    
    required_files = [
        "package.json",
        "next.config.js",
        "tailwind.config.js",
        "tsconfig.json"
    ]
    
    required_dirs = [
        "src",
        "src/components",
        "src/pages"
    ]
    
    all_valid = True
    
    # Check required files
    for file_name in required_files:
        file_path = f"{ui_dir}/{file_name}"
        if os.path.exists(file_path):
            print_success(f"Found {file_name}")
        else:
            print_error(f"Missing {file_name}")
            all_valid = False
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = f"{ui_dir}/{dir_name}"
        if os.path.isdir(dir_path):
            print_success(f"Found {dir_name}/")
        else:
            print_error(f"Missing {dir_name}/")
            all_valid = False
    
    # Check key components
    key_components = [
        "src/components/Layout.tsx",
        "src/components/Dashboard.tsx",
        "src/components/Login.tsx",
        "src/pages/index.tsx"
    ]
    
    for component in key_components:
        component_path = f"{ui_dir}/{component}"
        if os.path.exists(component_path):
            print_success(f"Found {component}")
        else:
            print_error(f"Missing {component}")
            all_valid = False
    
    return all_valid

def validate_documentation() -> bool:
    """Validate documentation"""
    print_header("Documentation Validation")
    
    docs_dir = "docs"
    if not os.path.isdir(docs_dir):
        print_error("Missing docs directory")
        return False
    
    required_docs = [
        "API.md",
        "DEPLOYMENT.md",
        "DEVELOPER_GUIDE.md"
    ]
    
    all_valid = True
    
    for doc in required_docs:
        doc_path = f"{docs_dir}/{doc}"
        if os.path.exists(doc_path):
            print_success(f"Found {doc}")
            
            # Check if file has content
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 100:  # Basic content check
                        print_success(f"  {doc} has content ({len(content)} chars)")
                    else:
                        print_warning(f"  {doc} seems empty or too short")
            except Exception as e:
                print_warning(f"  Could not read {doc}: {e}")
        else:
            print_error(f"Missing {doc}")
            all_valid = False
    
    return all_valid

def validate_configuration() -> bool:
    """Validate configuration files"""
    print_header("Configuration Validation")
    
    config_files = [
        (".env.example", "Environment template"),
        (".env.prod.example", "Production environment template"),
        ("docker-compose.yml", "Docker Compose development"),
        ("docker-compose.prod.yml", "Docker Compose production"),
        ("Makefile", "Build automation"),
        ("monitoring/prometheus.yml", "Prometheus configuration"),
        ("monitoring/alert_rules.yml", "Alert rules")
    ]
    
    all_valid = True
    
    for file_path, description in config_files:
        if os.path.exists(file_path):
            print_success(f"Found {description}: {file_path}")
            
            # Basic content validation
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        print_success(f"  {file_path} has content")
                    else:
                        print_warning(f"  {file_path} is empty")
            except Exception as e:
                print_warning(f"  Could not read {file_path}: {e}")
        else:
            print_error(f"Missing {description}: {file_path}")
            all_valid = False
    
    return all_valid

def validate_kubernetes() -> bool:
    """Validate Kubernetes manifests"""
    print_header("Kubernetes Manifests Validation")
    
    k8s_dir = "k8s"
    if not os.path.isdir(k8s_dir):
        print_error("Missing k8s directory")
        return False
    
    required_manifests = [
        "namespace.yaml",
        "configmap.yaml", 
        "secrets.yaml",
        "postgres.yaml",
        "redis.yaml",
        "ingress.yaml"
    ]
    
    all_valid = True
    
    for manifest in required_manifests:
        manifest_path = f"{k8s_dir}/{manifest}"
        if os.path.exists(manifest_path):
            print_success(f"Found {manifest}")
        else:
            print_error(f"Missing {manifest}")
            all_valid = False
    
    # Check services directory
    services_dir = f"{k8s_dir}/services"
    if os.path.isdir(services_dir):
        print_success("Found services/ directory")
        
        # Check for service manifests
        service_manifests = ["gateway.yaml", "auth.yaml"]
        for service_manifest in service_manifests:
            service_path = f"{services_dir}/{service_manifest}"
            if os.path.exists(service_path):
                print_success(f"  Found {service_manifest}")
            else:
                print_warning(f"  Missing {service_manifest}")
    else:
        print_error("Missing k8s/services/ directory")
        all_valid = False
    
    return all_valid

def run_basic_tests() -> bool:
    """Run basic Python syntax and import tests"""
    print_header("Basic Tests")
    
    all_valid = True
    
    # Test basic imports
    try:
        import fastapi
        print_success("FastAPI import successful")
    except ImportError as e:
        print_error(f"FastAPI import failed: {e}")
        all_valid = False
    
    try:
        import uvicorn
        print_success("Uvicorn import successful")
    except ImportError as e:
        print_error(f"Uvicorn import failed: {e}")
        all_valid = False
    
    try:
        import pytest
        print_success("Pytest import successful")
    except ImportError as e:
        print_error(f"Pytest import failed: {e}")
        all_valid = False
    
    # Test environment file
    if os.path.exists('.env'):
        print_success("Environment file (.env) exists")
        
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
                if 'DATABASE_URL' in env_content:
                    print_success("  DATABASE_URL found in .env")
                else:
                    print_warning("  DATABASE_URL not found in .env")
                
                if 'JWT_SECRET' in env_content:
                    print_success("  JWT_SECRET found in .env")
                else:
                    print_warning("  JWT_SECRET not found in .env")
        except Exception as e:
            print_error(f"Could not read .env file: {e}")
            all_valid = False
    else:
        print_warning("Environment file (.env) not found - using .env.example")
    
    return all_valid

def main():
    """Main validation function"""
    print_header("Lift OS Core - System Validation")
    print_info("Starting comprehensive system validation...")
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("File Structure", validate_file_structure),
        ("Services", validate_services),
        ("Modules", validate_modules),
        ("UI Shell", validate_ui_shell),
        ("Documentation", validate_documentation),
        ("Configuration", validate_configuration),
        ("Kubernetes", validate_kubernetes),
        ("Basic Tests", run_basic_tests)
    ]
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            validation_results.append((name, result))
        except Exception as e:
            print_error(f"Validation failed for {name}: {e}")
            validation_results.append((name, False))
    
    # Summary
    print_header("Validation Summary")
    
    passed = 0
    total = len(validation_results)
    
    for name, result in validation_results:
        if result:
            print_success(f"{name}: PASSED")
            passed += 1
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\nResults: {passed}/{total} validations passed")
    
    if passed == total:
        print_success("All validations passed! System is ready for deployment.")
        return True
    else:
        print_error(f"{total - passed} validation(s) failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)