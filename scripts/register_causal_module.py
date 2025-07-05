#!/usr/bin/env python3
"""
LiftOS Causal AI Module Registration Script
Registers the Causal AI module with the LiftOS registry service.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
REGISTRY_URL = "http://localhost:8005"
MODULE_CONFIG_PATH = Path("modules/causal/module.json")
MAX_RETRIES = 5
RETRY_DELAY = 2

def load_module_config():
    """Load module configuration from module.json"""
    try:
        with open(MODULE_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Module configuration not found at {MODULE_CONFIG_PATH}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in module configuration: {e}")
        return None

def wait_for_registry():
    """Wait for registry service to be available"""
    print("Waiting for registry service to be available...")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{REGISTRY_URL}/health", timeout=5)
            if response.status_code == 200:
                print("Registry service is available")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < MAX_RETRIES - 1:
            print(f"Registry not ready, retrying in {RETRY_DELAY} seconds... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
    
    print("ERROR: Registry service is not available after maximum retries")
    return False

def register_module(config):
    """Register the module with the registry service"""
    try:
        # Prepare registration payload
        registration_data = {
            "id": config["id"],
            "name": config["name"],
            "version": config["version"],
            "description": config["description"],
            "category": config["category"],
            "capabilities": config["capabilities"],
            "endpoints": config["endpoints"],
            "health_check": config["health_check"],
            "dependencies": config["dependencies"],
            "configuration": config["configuration"],
            "status": "active"
        }
        
        print(f"Registering module: {config['name']} (v{config['version']})")
        
        # Register with registry service
        response = requests.post(
            f"{REGISTRY_URL}/modules",
            json=registration_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            print("✓ Module registered successfully")
            return True
        elif response.status_code == 409:
            print("Module already exists, updating...")
            # Try to update existing module
            response = requests.put(
                f"{REGISTRY_URL}/modules/{config['id']}",
                json=registration_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                print("✓ Module updated successfully")
                return True
            else:
                print(f"ERROR: Failed to update module: {response.status_code} - {response.text}")
                return False
        else:
            print(f"ERROR: Failed to register module: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Network error during registration: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during registration: {e}")
        return False

def verify_registration(module_id):
    """Verify that the module was registered successfully"""
    try:
        response = requests.get(f"{REGISTRY_URL}/modules/{module_id}", timeout=5)
        if response.status_code == 200:
            module_data = response.json()
            print(f"✓ Module verification successful: {module_data['name']}")
            print(f"  Status: {module_data.get('status', 'unknown')}")
            print(f"  Capabilities: {len(module_data.get('capabilities', []))} defined")
            print(f"  Endpoints: {len(module_data.get('endpoints', []))} defined")
            return True
        else:
            print(f"ERROR: Module verification failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Could not verify registration: {e}")
        return False

def main():
    """Main registration process"""
    print("========================================")
    print("LiftOS Causal AI Module Registration")
    print("========================================")
    print()
    
    # Load module configuration
    config = load_module_config()
    if not config:
        sys.exit(1)
    
    # Wait for registry service
    if not wait_for_registry():
        sys.exit(1)
    
    # Register module
    if not register_module(config):
        sys.exit(1)
    
    # Verify registration
    verify_registration(config["id"])
    
    print()
    print("========================================")
    print("Causal AI Module Registration Complete!")
    print("========================================")
    print()
    print(f"Module ID: {config['id']}")
    print(f"Module Name: {config['name']}")
    print(f"Version: {config['version']}")
    print(f"Category: {config['category']}")
    print()
    print("The Causal AI module is now available in LiftOS!")

if __name__ == "__main__":
    main()