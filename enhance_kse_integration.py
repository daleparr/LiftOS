#!/usr/bin/env python3
"""
Enhance KSE Integration for All Services
Adds KSE-SDK imports and initialization to services with partial integration
"""

import os
import re
from pathlib import Path

class KSEIntegrationEnhancer:
    """Enhances services with KSE-SDK integration"""
    
    def __init__(self):
        self.services_dir = Path("services")
        self.partial_integration_services = [
            "bayesian-analysis",
            "billing", 
            "feedback",
            "impact-monitoring",
            "observability",
            "registry",
            "user-analytics"
        ]
        
    def add_kse_imports(self, file_path: Path) -> bool:
        """Add KSE-SDK imports to a service file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if KSE imports already exist
            if 'from shared.kse_sdk.client import LiftKSEClient' in content:
                print(f"  [SKIP] {file_path.name} already has KSE imports")
                return True
            
            # Find the import section
            import_pattern = r'(import sys\nimport os\nsys\.path\.append\(os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'.*?\)\))'
            match = re.search(import_pattern, content, re.MULTILINE | re.DOTALL)
            
            if match:
                # Add KSE imports after the sys.path.append
                kse_imports = """
# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain
"""
                
                new_content = content.replace(
                    match.group(1),
                    match.group(1) + kse_imports
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  [ADDED] KSE imports to {file_path.name}")
                return True
            else:
                print(f"  [WARN] Could not find import section in {file_path.name}")
                return False
                
        except Exception as e:
            print(f"  [ERROR] Failed to enhance {file_path.name}: {e}")
            return False
    
    def add_kse_client_initialization(self, file_path: Path) -> bool:
        """Add KSE client initialization to a service file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if KSE client already exists
            if 'kse_client = LiftKSEClient()' in content or 'kse_client = None' in content:
                print(f"  [SKIP] {file_path.name} already has KSE client")
                return True
            
            # Find FastAPI app initialization or class initialization
            patterns = [
                r'(app = FastAPI\()',
                r'(class \w+Engine:)',
                r'(class \w+Service:)',
                r'(# Configure logging\nlogging\.basicConfig)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    # Add KSE client before the match
                    kse_init = """
# KSE Client for intelligence integration
kse_client = None

async def initialize_kse_client():
    \"\"\"Initialize KSE client for intelligence integration\"\"\"
    global kse_client
    try:
        kse_client = LiftKSEClient()
        print("KSE Client initialized successfully")
        return True
    except Exception as e:
        print(f"KSE Client initialization failed: {e}")
        kse_client = None
        return False

"""
                    
                    new_content = content.replace(
                        match.group(1),
                        kse_init + match.group(1)
                    )
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print(f"  [ADDED] KSE client initialization to {file_path.name}")
                    return True
            
            print(f"  [WARN] Could not find suitable location for KSE client in {file_path.name}")
            return False
            
        except Exception as e:
            print(f"  [ERROR] Failed to add KSE client to {file_path.name}: {e}")
            return False
    
    def enhance_service(self, service_name: str) -> bool:
        """Enhance a single service with KSE integration"""
        service_dir = self.services_dir / service_name
        app_file = service_dir / "app.py"
        
        if not app_file.exists():
            print(f"  [SKIP] {service_name}/app.py not found")
            return False
        
        print(f"\n[*] Enhancing {service_name}...")
        
        # Add imports
        imports_added = self.add_kse_imports(app_file)
        
        # Add client initialization
        client_added = self.add_kse_client_initialization(app_file)
        
        return imports_added and client_added
    
    def enhance_all_services(self):
        """Enhance all services with partial KSE integration"""
        print("="*60)
        print("KSE Integration Enhancement Tool")
        print("="*60)
        
        enhanced_count = 0
        total_count = len(self.partial_integration_services)
        
        for service_name in self.partial_integration_services:
            if self.enhance_service(service_name):
                enhanced_count += 1
        
        print(f"\n{'='*60}")
        print(f"Enhancement Summary: {enhanced_count}/{total_count} services enhanced")
        print(f"{'='*60}")
        
        return enhanced_count == total_count

if __name__ == "__main__":
    enhancer = KSEIntegrationEnhancer()
    success = enhancer.enhance_all_services()
    
    if success:
        print("\n[SUCCESS] All services enhanced with KSE integration!")
    else:
        print("\n[PARTIAL] Some services could not be enhanced automatically")
    
    print("\nNext steps:")
    print("1. Run the microservice integration diagnostic to verify improvements")
    print("2. Test individual services to ensure they start correctly")
    print("3. Verify KSE client initialization in service logs")