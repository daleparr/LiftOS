#!/usr/bin/env python3
"""
Deployment Script for Tier 1 API Connectors
Validates setup, runs tests, and provides deployment guidance
"""
import asyncio
import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class Tier1ConnectorDeployment:
    """Deployment manager for Tier 1 connectors"""
    
    def __init__(self):
        self.project_root = project_root
        self.service_root = Path(__file__).parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    def print_status(self, message: str, status: str = "INFO"):
        """Print status message with formatting"""
        colors = {
            "INFO": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"     # Reset
        }
        
        color = colors.get(status, colors["INFO"])
        reset = colors["RESET"]
        print(f"{color}[{status}]{reset} {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        self.print_header("Python Version Check")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "SUCCESS")
            return True
        else:
            error_msg = f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+"
            self.print_status(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        self.print_header("Dependency Check")
        
        required_packages = [
            "fastapi", "uvicorn", "httpx", "pydantic", "asyncio",
            "pytest", "pytest-asyncio", "boto3", "cryptography"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                self.print_status(f"✓ {package}", "SUCCESS")
            except ImportError:
                self.print_status(f"✗ {package} - Missing", "ERROR")
                missing_packages.append(package)
        
        if missing_packages:
            error_msg = f"Missing packages: {', '.join(missing_packages)}"
            self.errors.append(error_msg)
            self.print_status("Run: pip install -r requirements_tier1.txt", "WARNING")
            return False
        
        return True
    
    def check_file_structure(self) -> bool:
        """Check if all required files exist"""
        self.print_header("File Structure Check")
        
        required_files = [
            "app.py",
            "connectors/shopify_connector.py",
            "connectors/woocommerce_connector.py", 
            "connectors/amazon_connector.py",
            "test_tier1_connectors.py",
            "TIER1_CONNECTORS_README.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.service_root / file_path
            if full_path.exists():
                self.print_status(f"✓ {file_path}", "SUCCESS")
            else:
                self.print_status(f"✗ {file_path} - Missing", "ERROR")
                missing_files.append(file_path)
        
        if missing_files:
            error_msg = f"Missing files: {', '.join(missing_files)}"
            self.errors.append(error_msg)
            return False
        
        return True
    
    def check_shared_models(self) -> bool:
        """Check if shared models are accessible"""
        self.print_header("Shared Models Check")
        
        try:
            from shared.models.marketing import DataSource
            from shared.models.causal_marketing import CausalMarketingData
            
            # Check if new data sources are available
            sources = [source.value for source in DataSource]
            required_sources = ["shopify", "woocommerce", "amazon_seller_central"]
            
            missing_sources = [s for s in required_sources if s not in sources]
            if missing_sources:
                error_msg = f"Missing DataSource enums: {', '.join(missing_sources)}"
                self.print_status(error_msg, "ERROR")
                self.errors.append(error_msg)
                return False
            
            self.print_status("✓ All DataSource enums present", "SUCCESS")
            self.print_status("✓ CausalMarketingData model accessible", "SUCCESS")
            return True
            
        except ImportError as e:
            error_msg = f"Cannot import shared models: {str(e)}"
            self.print_status(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False
    
    def validate_connector_imports(self) -> bool:
        """Validate that connectors can be imported"""
        self.print_header("Connector Import Validation")
        
        connectors = [
            ("connectors.shopify_connector", "ShopifyConnector"),
            ("connectors.woocommerce_connector", "WooCommerceConnector"),
            ("connectors.amazon_connector", "AmazonConnector")
        ]
        
        import_errors = []
        for module_name, class_name in connectors:
            try:
                # Change to service directory for imports
                original_cwd = os.getcwd()
                os.chdir(self.service_root)
                
                module = importlib.import_module(module_name)
                connector_class = getattr(module, class_name)
                self.print_status(f"✓ {class_name} imported successfully", "SUCCESS")
                
                os.chdir(original_cwd)
                
            except Exception as e:
                error_msg = f"✗ {class_name} import failed: {str(e)}"
                self.print_status(error_msg, "ERROR")
                import_errors.append(error_msg)
                
                if 'original_cwd' in locals():
                    os.chdir(original_cwd)
        
        if import_errors:
            self.errors.extend(import_errors)
            return False
        
        return True
    
    async def run_basic_tests(self) -> bool:
        """Run basic functionality tests"""
        self.print_header("Basic Functionality Tests")
        
        try:
            # Change to service directory
            original_cwd = os.getcwd()
            os.chdir(self.service_root)
            
            # Run pytest with basic tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "test_tier1_connectors.py::TestKSEIntegration::test_neural_content_generation",
                "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                self.print_status("✓ Basic tests passed", "SUCCESS")
                return True
            else:
                error_msg = f"Tests failed: {result.stderr}"
                self.print_status(error_msg, "ERROR")
                self.errors.append(error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "Tests timed out after 60 seconds"
            self.print_status(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            self.print_status(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables"""
        self.print_header("Environment Variables Check")
        
        recommended_vars = [
            ("MEMORY_SERVICE_URL", "http://localhost:8003"),
            ("KSE_SERVICE_URL", "http://localhost:8004"),
            ("LOG_LEVEL", "INFO")
        ]
        
        missing_vars = []
        for var_name, default_value in recommended_vars:
            value = os.getenv(var_name)
            if value:
                self.print_status(f"✓ {var_name}={value}", "SUCCESS")
            else:
                self.print_status(f"⚠ {var_name} not set (default: {default_value})", "WARNING")
                self.warnings.append(f"Consider setting {var_name}={default_value}")
        
        return True  # Environment variables are optional for basic functionality
    
    def generate_deployment_summary(self) -> Dict[str, any]:
        """Generate deployment summary"""
        return {
            "status": "ready" if not self.errors else "failed",
            "errors": self.errors,
            "warnings": self.warnings,
            "connectors": ["Shopify", "WooCommerce", "Amazon Seller Central"],
            "features": [
                "KSE Universal Substrate Integration",
                "Causal Marketing Data Transformation", 
                "Treatment Assignment Logic",
                "Confounder Detection",
                "Cross-Platform Attribution"
            ]
        }
    
    def print_deployment_instructions(self):
        """Print deployment instructions"""
        self.print_header("Deployment Instructions")
        
        if not self.errors:
            self.print_status("✓ All checks passed! Ready for deployment", "SUCCESS")
            print("\nNext steps:")
            print("1. Start the Memory Service (port 8003)")
            print("2. Start the KSE Service (port 8004)")  
            print("3. Configure platform credentials in credential manager")
            print("4. Start the Data Ingestion Service:")
            print("   cd services/data-ingestion")
            print("   python app.py")
            print("\n5. Test the API endpoints:")
            print("   GET  http://localhost:8006/health")
            print("   GET  http://localhost:8006/docs")
            print("   POST http://localhost:8006/sync/start")
            
        else:
            self.print_status("✗ Deployment checks failed", "ERROR")
            print("\nErrors to fix:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
    
    async def run_full_deployment_check(self) -> bool:
        """Run complete deployment validation"""
        self.print_header("LiftOS Tier 1 Connectors - Deployment Validation")
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("File Structure", self.check_file_structure),
            ("Shared Models", self.check_shared_models),
            ("Connector Imports", self.validate_connector_imports),
            ("Environment Variables", self.check_environment_variables),
            ("Basic Tests", self.run_basic_tests)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                error_msg = f"{check_name} check failed: {str(e)}"
                self.print_status(error_msg, "ERROR")
                self.errors.append(error_msg)
                all_passed = False
        
        # Generate and print summary
        summary = self.generate_deployment_summary()
        self.print_deployment_instructions()
        
        return all_passed


async def main():
    """Main deployment validation function"""
    deployment = Tier1ConnectorDeployment()
    
    try:
        success = await deployment.run_full_deployment_check()
        
        if success:
            print(f"\n🎉 Tier 1 Connectors deployment validation completed successfully!")
            print("Ready to process Shopify, WooCommerce, and Amazon Seller Central data")
            print("with full KSE Universal Substrate integration.")
            return 0
        else:
            print(f"\n❌ Deployment validation failed. Please fix the errors above.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Deployment validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error during deployment validation: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)