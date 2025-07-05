#!/usr/bin/env python3
"""
LiftOS LLM Module Registration Script
Registers the LLM module with the LiftOS registry service.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
REGISTRY_URL = "http://localhost:8005"
MODULE_CONFIG_PATH = Path("modules/llm/module.json")
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
            "ui_components": config.get("ui_components", []),
            "integrations": config.get("integrations", []),
            "prompt_templates": config.get("prompt_templates", []),
            "evaluation_metrics": config.get("evaluation_metrics", []),
            "resource_requirements": config.get("resource_requirements", {}),
            "permissions": config.get("permissions", []),
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
            print(f"  Integrations: {len(module_data.get('integrations', []))} providers")
            print(f"  Templates: {len(module_data.get('prompt_templates', []))} available")
            print(f"  Metrics: {len(module_data.get('evaluation_metrics', []))} supported")
            return True
        else:
            print(f"ERROR: Module verification failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Could not verify registration: {e}")
        return False

def test_module_endpoints(module_id):
    """Test basic module endpoints"""
    print("Testing module endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8009/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"⚠ Health endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"⚠ Health endpoint test failed: {e}")
    
    # Test templates endpoint
    try:
        response = requests.get("http://localhost:8009/api/v1/prompts/templates", timeout=5)
        if response.status_code == 200:
            templates = response.json()
            print(f"✓ Templates endpoint working ({len(templates.get('templates', []))} templates)")
        else:
            print(f"⚠ Templates endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"⚠ Templates endpoint test failed: {e}")
    
    # Test leaderboard endpoint
    try:
        response = requests.get("http://localhost:8009/api/v1/models/leaderboard", timeout=5)
        if response.status_code == 200:
            print("✓ Leaderboard endpoint working")
        else:
            print(f"⚠ Leaderboard endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"⚠ Leaderboard endpoint test failed: {e}")

def main():
    """Main registration process"""
    print("========================================")
    print("LiftOS LLM Module Registration")
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
    
    # Test module endpoints
    test_module_endpoints(config["id"])
    
    print()
    print("========================================")
    print("LLM Module Registration Complete!")
    print("========================================")
    print()
    print(f"Module ID: {config['id']}")
    print(f"Module Name: {config['name']}")
    print(f"Version: {config['version']}")
    print(f"Category: {config['category']}")
    print()
    print("Available capabilities:")
    for capability in config.get('capabilities', []):
        print(f"  • {capability}")
    print()
    print("Supported providers:")
    for integration in config.get('integrations', []):
        print(f"  • {integration['provider']} ({integration['type']})")
    print()
    print("Prompt templates:")
    for template in config.get('prompt_templates', []):
        print(f"  • {template['name']}: {template['description']}")
    print()
    print("Evaluation metrics:")
    for metric in config.get('evaluation_metrics', []):
        print(f"  • {metric['name']}: {metric['description']}")
    print()
    print("The LLM module is now available in LiftOS!")
    print("Access it via: http://localhost:8000/modules/llm/")

if __name__ == "__main__":
    main()