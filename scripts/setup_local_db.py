"""
Local database setup for development
Creates SQLite database and in-memory cache for testing
"""

import os
import sys
import asyncio
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.database.models import Base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalDatabaseSetup:
    """Setup local SQLite database for development"""
    
    def __init__(self):
        self.db_path = project_root / "data" / "lift_os_dev.db"
        self.db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.engine = None
        self.session_factory = None
    
    async def setup(self):
        """Setup local database"""
        try:
            # Create data directory
            self.db_path.parent.mkdir(exist_ok=True)
            
            # Create async engine
            self.engine = create_async_engine(
                self.db_url,
                echo=True,
                future=True
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info(f"Database created at: {self.db_path}")
            
            # Insert sample data
            await self.insert_sample_data()
            
            logger.info("Local database setup complete!")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    async def insert_sample_data(self):
        """Insert sample data for testing"""
        from shared.database.models import User, Module, BillingAccount
        from datetime import datetime, timezone
        import uuid
        import bcrypt
        
        async with self.session_factory() as session:
            try:
                # Create sample user
                hashed_password = bcrypt.hashpw("testpassword".encode(), bcrypt.gensalt()).decode()
                
                sample_user = User(
                    id=uuid.uuid4(),
                    email="test@lift.com",
                    username="testuser",
                    hashed_password=hashed_password,
                    full_name="Test User",
                    is_active=True,
                    is_verified=True,
                    is_superuser=True
                )
                session.add(sample_user)
                await session.flush()  # Get the user ID
                
                # Create sample modules
                sample_modules = [
                    Module(
                        name="auth",
                        display_name="Authentication Service",
                        description="Core authentication and authorization service",
                        version="1.0.0",
                        service_url="http://localhost:8001",
                        port=8001,
                        status="active",
                        config={"jwt_expiration": 24},
                        endpoints=["/login", "/register", "/verify", "/refresh"]
                    ),
                    Module(
                        name="memory",
                        display_name="KSE Memory Service",
                        description="Knowledge storage and retrieval service",
                        version="1.0.0",
                        service_url="http://localhost:8002",
                        port=8002,
                        status="active",
                        config={"max_contexts": 100},
                        endpoints=["/store", "/retrieve", "/search", "/contexts"]
                    ),
                    Module(
                        name="registry",
                        display_name="Module Registry",
                        description="Service discovery and module management",
                        version="1.0.0",
                        service_url="http://localhost:8004",
                        port=8004,
                        status="active",
                        config={"auto_discovery": True},
                        endpoints=["/modules", "/register", "/health"]
                    )
                ]
                
                for module in sample_modules:
                    module.registered_by = sample_user.id
                    session.add(module)
                
                # Create sample billing account
                billing_account = BillingAccount(
                    user_id=sample_user.id,
                    plan_type="pro",
                    billing_email="test@lift.com",
                    status="active",
                    current_usage={"api_calls": 150, "storage_mb": 500},
                    usage_limits={"api_calls": 10000, "storage_mb": 5000}
                )
                session.add(billing_account)
                
                await session.commit()
                logger.info("Sample data inserted successfully")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to insert sample data: {e}")
                raise
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            await self.engine.dispose()

async def main():
    """Main setup function"""
    setup = LocalDatabaseSetup()
    try:
        await setup.setup()
        print("\n" + "="*50)
        print("LOCAL DATABASE SETUP COMPLETE")
        print("="*50)
        print(f"Database file: {setup.db_path}")
        print(f"Database URL: {setup.db_url}")
        print("\nSample data created:")
        print("- User: test@lift.com / testpassword")
        print("- Modules: auth, memory, registry")
        print("- Billing account: pro plan")
        print("\nYou can now run services with local database!")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)
    finally:
        await setup.cleanup()

if __name__ == "__main__":
    asyncio.run(main())