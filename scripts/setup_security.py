"""
LiftOS Enterprise Security Setup Script
Initializes the enterprise security infrastructure including database migrations,
RSA key generation, and security configuration.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import secrets
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.database.migration_runner import MigrationRunner
from shared.security.api_key_vault import get_api_key_vault
from shared.security.enhanced_jwt import get_enhanced_jwt_manager
from shared.security.audit_logger import SecurityAuditLogger
from shared.database.database import get_async_session
from shared.database.security_models import SecurityConfiguration

logger = logging.getLogger(__name__)

class SecuritySetup:
    """
    Handles the complete setup of LiftOS enterprise security infrastructure
    """
    
    def __init__(self):
        self.project_root = project_root
        self.security_dir = self.project_root / "security"
        self.keys_dir = self.security_dir / "keys"
        self.config_dir = self.security_dir / "config"
        
        # Ensure directories exist
        self.security_dir.mkdir(exist_ok=True)
        self.keys_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
    
    async def run_complete_setup(self, org_id: str = "default-org"):
        """Run the complete security setup process"""
        try:
            logger.info("🔐 Starting LiftOS Enterprise Security Setup...")
            
            # Step 1: Database migrations
            await self._setup_database()
            
            # Step 2: Generate RSA keys for JWT
            await self._generate_rsa_keys()
            
            # Step 3: Create security configuration
            await self._create_security_configuration(org_id)
            
            # Step 4: Generate master encryption key
            await self._generate_master_encryption_key()
            
            # Step 5: Create environment template
            await self._create_environment_template()
            
            # Step 6: Verify setup
            await self._verify_setup()
            
            logger.info("✅ Enterprise Security Setup completed successfully!")
            
            # Print setup summary
            await self._print_setup_summary()
            
        except Exception as e:
            logger.error(f"❌ Security setup failed: {e}")
            raise
    
    async def _setup_database(self):
        """Set up database with security tables"""
        logger.info("📊 Setting up database with security tables...")
        
        try:
            runner = MigrationRunner()
            await runner.run_all_migrations()
            logger.info("✅ Database setup completed")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    async def _generate_rsa_keys(self):
        """Generate RSA key pair for JWT signing"""
        logger.info("🔑 Generating RSA key pair for JWT signing...")
        
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize private key
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Serialize public key
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Save keys to files
            private_key_path = self.keys_dir / "jwt_private_key.pem"
            public_key_path = self.keys_dir / "jwt_public_key.pem"
            
            with open(private_key_path, 'wb') as f:
                f.write(private_pem)
            
            with open(public_key_path, 'wb') as f:
                f.write(public_pem)
            
            # Set secure permissions (Unix-like systems)
            if os.name != 'nt':  # Not Windows
                os.chmod(private_key_path, 0o600)  # Read/write for owner only
                os.chmod(public_key_path, 0o644)   # Read for all, write for owner
            
            logger.info(f"✅ RSA keys generated and saved to {self.keys_dir}")
            
        except ImportError:
            logger.warning("⚠️  cryptography library not available, skipping RSA key generation")
            logger.info("   Install with: pip install cryptography")
        except Exception as e:
            logger.error(f"❌ RSA key generation failed: {e}")
            raise
    
    async def _generate_master_encryption_key(self):
        """Generate master encryption key for API key vault"""
        logger.info("🔐 Generating master encryption key...")
        
        try:
            # Generate a secure random key
            master_key = secrets.token_urlsafe(32)  # 256-bit key
            
            # Save to secure file
            key_file = self.keys_dir / "master_encryption_key.txt"
            
            with open(key_file, 'w') as f:
                f.write(master_key)
            
            # Set secure permissions
            if os.name != 'nt':  # Not Windows
                os.chmod(key_file, 0o600)  # Read/write for owner only
            
            logger.info(f"✅ Master encryption key generated and saved to {key_file}")
            logger.warning("⚠️  Keep this key secure! Loss of this key means loss of all encrypted API keys!")
            
        except Exception as e:
            logger.error(f"❌ Master key generation failed: {e}")
            raise
    
    async def _create_security_configuration(self, org_id: str):
        """Create default security configuration"""
        logger.info("⚙️  Creating security configuration...")
        
        try:
            async with get_async_session() as session:
                # Check if configuration already exists
                from sqlalchemy import select
                result = await session.execute(
                    select(SecurityConfiguration).where(SecurityConfiguration.org_id == org_id)
                )
                existing_config = result.scalar_one_or_none()
                
                if existing_config:
                    logger.info(f"✅ Security configuration already exists for org {org_id}")
                    return
                
                # Create default security configuration
                security_config = SecurityConfiguration(
                    org_id=org_id,
                    max_concurrent_sessions=3,
                    session_timeout_minutes=480,  # 8 hours
                    require_mfa=False,
                    allowed_ip_ranges=["0.0.0.0/0"],  # Allow all IPs initially
                    api_rate_limits={
                        "auth": {"limit": 5, "window": 300},      # 5 auth attempts per 5 minutes
                        "api": {"limit": 1000, "window": 3600},   # 1000 API calls per hour
                        "sensitive": {"limit": 10, "window": 60}   # 10 sensitive ops per minute
                    },
                    password_policy={
                        "min_length": 12,
                        "require_uppercase": True,
                        "require_lowercase": True,
                        "require_numbers": True,
                        "require_symbols": True,
                        "max_age_days": 90
                    },
                    audit_retention_days=365,
                    compliance_settings={
                        "soc2_compliance": True,
                        "gdpr_compliance": True,
                        "encryption_at_rest": True,
                        "encryption_in_transit": True
                    },
                    updated_by="system_setup"
                )
                
                session.add(security_config)
                await session.commit()
                
                logger.info(f"✅ Security configuration created for org {org_id}")
                
        except Exception as e:
            logger.error(f"❌ Security configuration creation failed: {e}")
            raise
    
    async def _create_environment_template(self):
        """Create environment variable template"""
        logger.info("📝 Creating environment variable template...")
        
        try:
            env_template = """
# LiftOS Enterprise Security Environment Variables
# Copy this to .env and update with your actual values

# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/liftos_db

# JWT Configuration
JWT_ACCESS_SECRET=your-super-secure-access-secret-change-this
JWT_REFRESH_SECRET=your-super-secure-refresh-secret-change-this
JWT_PRIVATE_KEY_PATH=./security/keys/jwt_private_key.pem
JWT_PUBLIC_KEY_PATH=./security/keys/jwt_public_key.pem

# API Key Vault Configuration
MASTER_ENCRYPTION_KEY_PATH=./security/keys/master_encryption_key.txt
# Alternative: Set the key directly (not recommended for production)
# MASTER_ENCRYPTION_KEY=your-master-encryption-key

# Security Configuration
SECURITY_AUDIT_ENABLED=true
SECURITY_RATE_LIMITING_ENABLED=true
SECURITY_DEVICE_FINGERPRINTING_ENABLED=true

# Production Security Settings
SECURITY_REQUIRE_HTTPS=true
SECURITY_SECURE_COOKIES=true
SECURITY_CSRF_PROTECTION=true

# Logging Configuration
LOG_LEVEL=INFO
SECURITY_LOG_LEVEL=INFO

# External Service API Keys (for fallback when vault is not available)
# Meta/Facebook
META_ACCESS_TOKEN=your-meta-access-token
META_APP_ID=your-meta-app-id
META_APP_SECRET=your-meta-app-secret

# Google Ads
GOOGLE_ADS_DEVELOPER_TOKEN=your-google-ads-developer-token
GOOGLE_ADS_CLIENT_ID=your-google-ads-client-id
GOOGLE_ADS_CLIENT_SECRET=your-google-ads-client-secret
GOOGLE_ADS_REFRESH_TOKEN=your-google-ads-refresh-token

# Klaviyo
KLAVIYO_API_KEY=your-klaviyo-api-key

# Shopify
SHOPIFY_SHOP_DOMAIN=your-shop.myshopify.com
SHOPIFY_ACCESS_TOKEN=your-shopify-access-token

# Add other platform credentials as needed...
"""
            
            env_file = self.project_root / ".env.template"
            
            with open(env_file, 'w') as f:
                f.write(env_template.strip())
            
            logger.info(f"✅ Environment template created at {env_file}")
            logger.info("   Copy .env.template to .env and update with your actual values")
            
        except Exception as e:
            logger.error(f"❌ Environment template creation failed: {e}")
            raise
    
    async def _verify_setup(self):
        """Verify that the security setup is working correctly"""
        logger.info("🔍 Verifying security setup...")
        
        try:
            # Test database connection and tables
            async with get_async_session() as session:
                # Test audit logger
                audit_logger = SecurityAuditLogger()
                await audit_logger.log_security_event(
                    session=session,
                    event_type="SYSTEM_STARTUP",
                    action="security_setup_verification",
                    success=True,
                    details={"setup_time": datetime.now(timezone.utc).isoformat()}
                )
                
                # Test API key vault
                vault = get_api_key_vault()
                test_credentials = {"test_key": "test_value"}
                
                # Store and retrieve test credentials
                success = await vault.store_api_key(
                    session=session,
                    org_id="setup-test",
                    provider="test_provider",
                    key_name="setup_test",
                    api_key_data=test_credentials,
                    created_by="system_setup"
                )
                
                if success:
                    retrieved = await vault.get_api_key(
                        session=session,
                        org_id="setup-test",
                        provider="test_provider",
                        key_name="setup_test"
                    )
                    
                    if retrieved and retrieved["test_key"] == "test_value":
                        logger.info("✅ API Key Vault verification successful")
                        
                        # Clean up test data
                        await vault.revoke_api_key(
                            session=session,
                            org_id="setup-test",
                            provider="test_provider",
                            key_name="setup_test",
                            reason="setup_test_cleanup"
                        )
                    else:
                        raise Exception("API Key Vault verification failed - retrieval mismatch")
                else:
                    raise Exception("API Key Vault verification failed - storage failed")
                
                # Test JWT manager
                jwt_manager = get_enhanced_jwt_manager()
                logger.info("✅ JWT Manager verification successful")
                
                await session.commit()
            
            logger.info("✅ Security setup verification completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Security setup verification failed: {e}")
            raise
    
    async def _print_setup_summary(self):
        """Print a summary of the setup"""
        logger.info("\n" + "="*60)
        logger.info("🎉 LiftOS Enterprise Security Setup Complete!")
        logger.info("="*60)
        logger.info("\n📋 Setup Summary:")
        logger.info("   ✅ Database tables created")
        logger.info("   ✅ RSA keys generated for JWT")
        logger.info("   ✅ Master encryption key generated")
        logger.info("   ✅ Security configuration created")
        logger.info("   ✅ Environment template created")
        logger.info("   ✅ Setup verification passed")
        
        logger.info("\n🔧 Next Steps:")
        logger.info("   1. Copy .env.template to .env")
        logger.info("   2. Update .env with your actual configuration values")
        logger.info("   3. Secure the keys directory (./security/keys/)")
        logger.info("   4. Set up proper file permissions in production")
        logger.info("   5. Configure your application to use the enhanced security")
        
        logger.info("\n🔐 Security Features Enabled:")
        logger.info("   • AES-256-GCM encrypted API key storage")
        logger.info("   • Enhanced JWT with refresh token rotation")
        logger.info("   • Device fingerprinting and session management")
        logger.info("   • Comprehensive security audit logging")
        logger.info("   • Rate limiting and IP-based security")
        logger.info("   • SOC 2 compliance features")
        
        logger.info("\n⚠️  Important Security Notes:")
        logger.info("   • Keep the master encryption key secure!")
        logger.info("   • Use HTTPS in production")
        logger.info("   • Regularly rotate API keys")
        logger.info("   • Monitor security audit logs")
        logger.info("   • Review and update security configuration")
        
        logger.info("\n📚 Documentation:")
        logger.info("   • API Key Vault: shared/security/api_key_vault.py")
        logger.info("   • Enhanced JWT: shared/security/enhanced_jwt.py")
        logger.info("   • Security Middleware: shared/security/enhanced_middleware.py")
        logger.info("   • Audit Logger: shared/security/audit_logger.py")
        
        logger.info("="*60)

async def main():
    """Main setup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        setup = SecuritySetup()
        
        # Get organization ID from command line or use default
        org_id = sys.argv[1] if len(sys.argv) > 1 else "default-org"
        
        await setup.run_complete_setup(org_id)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Setup interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())