#!/usr/bin/env python3
"""
Script to register the Surfacing module with the LiftOS Core registry
"""
import asyncio
import json
import sys
import os
from pathlib import Path
import httpx

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

REGISTRY_URL = os.getenv("REGISTRY_SERVICE_URL", "http://localhost:8005")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")

async def register_surfacing_module():
    """Register the surfacing module with the registry service"""
    
    # Load module configuration
    module_config_path = project_root / "modules" / "surfacing" / "module.json"
    
    if not module_config_path.exists():
        print(f"❌ Module configuration not found: {module_config_path}")
        return False
    
    with open(module_config_path, 'r') as f:
        module_config = json.load(f)
    
    # Prepare registration request
    registration_data = {
        "name": module_config["name"],
        "version": module_config["version"],
        "base_url": module_config["base_url"],
        "health_endpoint": module_config["health_endpoint"],
        "api_prefix": module_config["api_prefix"],
        "features": module_config["features"],
        "permissions": module_config["permissions"],
        "memory_requirements": module_config["memory_requirements"],
        "ui_components": module_config["ui_components"],
        "metadata": module_config["metadata"]
    }
    
    print(f"🚀 Registering Surfacing module...")
    print(f"   Module: {registration_data['name']} v{registration_data['version']}")
    print(f"   URL: {registration_data['base_url']}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Check if registry is available
            print(f"🔍 Checking registry service at {REGISTRY_URL}...")
            health_response = await client.get(f"{REGISTRY_URL}/health")
            
            if health_response.status_code != 200:
                print(f"❌ Registry service not healthy: {health_response.status_code}")
                return False
            
            print("✅ Registry service is healthy")
            
            # Check if surfacing module is running
            print(f"🔍 Checking surfacing module at {registration_data['base_url']}...")
            try:
                module_health = await client.get(f"{registration_data['base_url']}/health")
                if module_health.status_code == 200:
                    print("✅ Surfacing module is healthy")
                else:
                    print(f"⚠️  Surfacing module health check failed: {module_health.status_code}")
            except Exception as e:
                print(f"⚠️  Could not reach surfacing module: {e}")
            
            # Register the module
            print("📝 Registering module with registry...")
            
            # For development, we'll use mock headers since auth might not be fully set up
            headers = {
                "X-User-ID": "dev-user",
                "X-Org-ID": "dev-org", 
                "X-User-Roles": "admin,developer"
            }
            
            response = await client.post(
                f"{REGISTRY_URL}/modules",
                json=registration_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                module_id = result["data"]["module_id"]
                print(f"✅ Successfully registered surfacing module!")
                print(f"   Module ID: {module_id}")
                print(f"   Status: {result['data']['health']['status']}")
                
                # Test module access through gateway
                print(f"🔍 Testing module access through gateway...")
                try:
                    gateway_response = await client.get(
                        f"{GATEWAY_URL}/modules/{module_config['module_id']}/",
                        headers=headers
                    )
                    if gateway_response.status_code == 200:
                        print("✅ Module accessible through gateway")
                    else:
                        print(f"⚠️  Gateway access test failed: {gateway_response.status_code}")
                except Exception as e:
                    print(f"⚠️  Gateway access test error: {e}")
                
                return True
            else:
                print(f"❌ Registration failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False

async def check_module_status():
    """Check the status of the surfacing module"""
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Check registry for surfacing module
            print("🔍 Checking surfacing module status...")
            
            headers = {
                "X-User-ID": "dev-user",
                "X-Org-ID": "dev-org", 
                "X-User-Roles": "admin,developer"
            }
            
            response = await client.get(
                f"{REGISTRY_URL}/modules",
                headers=headers
            )
            
            if response.status_code == 200:
                modules = response.json()["data"]["items"]
                surfacing_modules = [m for m in modules if m["name"] == "Surfacing Module"]
                
                if surfacing_modules:
                    module = surfacing_modules[0]
                    print(f"✅ Found surfacing module:")
                    print(f"   ID: {module['id']}")
                    print(f"   Status: {module['status']}")
                    print(f"   Health: {module.get('health', {}).get('status', 'unknown')}")
                    print(f"   URL: {module['base_url']}")
                else:
                    print("❌ Surfacing module not found in registry")
            else:
                print(f"❌ Could not check registry: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Status check error: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Surfacing module registration")
    parser.add_argument("action", choices=["register", "status"], 
                       help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "register":
        success = asyncio.run(register_surfacing_module())
        sys.exit(0 if success else 1)
    elif args.action == "status":
        asyncio.run(check_module_status())

if __name__ == "__main__":
    main()