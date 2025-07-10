"""
Database Migration Runner for LiftOS Security
Handles creation and execution of security-related database migrations
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.database.database import get_async_session, get_database_url
from shared.database.security_models import Base

logger = logging.getLogger(__name__)

class MigrationRunner:
    """
    Handles database migrations for security features
    """
    
    def __init__(self):
        self.database_url = get_database_url()
        self.engine = create_async_engine(self.database_url)
        self.migrations_table = "schema_migrations"
        
    async def initialize_migrations_table(self):
        """Create the migrations tracking table if it doesn't exist"""
        try:
            async with self.engine.begin() as conn:
                # Check if migrations table exists
                result = await conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": self.migrations_table})
                
                table_exists = result.scalar()
                
                if not table_exists:
                    # Create migrations table
                    await conn.execute(text(f"""
                        CREATE TABLE {self.migrations_table} (
                            id SERIAL PRIMARY KEY,
                            version VARCHAR(255) NOT NULL UNIQUE,
                            description TEXT,
                            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            execution_time_ms INTEGER,
                            checksum VARCHAR(64)
                        )
                    """))
                    
                    logger.info(f"Created {self.migrations_table} table")
                else:
                    logger.info(f"{self.migrations_table} table already exists")
                    
        except Exception as e:
            logger.error(f"Failed to initialize migrations table: {e}")
            raise
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(f"""
                    SELECT version FROM {self.migrations_table} 
                    ORDER BY applied_at
                """))
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    async def record_migration(
        self, 
        version: str, 
        description: str, 
        execution_time_ms: int,
        checksum: str
    ):
        """Record a successfully applied migration"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text(f"""
                    INSERT INTO {self.migrations_table} 
                    (version, description, execution_time_ms, checksum)
                    VALUES (:version, :description, :execution_time_ms, :checksum)
                """), {
                    "version": version,
                    "description": description,
                    "execution_time_ms": execution_time_ms,
                    "checksum": checksum
                })
                
                logger.info(f"Recorded migration {version}")
                
        except Exception as e:
            logger.error(f"Failed to record migration {version}: {e}")
            raise
    
    async def create_security_tables(self):
        """Create all security tables using SQLAlchemy models"""
        try:
            start_time = datetime.now()
            
            async with self.engine.begin() as conn:
                # Create all tables defined in security models
                await conn.run_sync(Base.metadata.create_all)
                
                logger.info("Created security tables using SQLAlchemy models")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record this as a migration
            await self.record_migration(
                version="001_security_tables_sqlalchemy",
                description="Create security tables using SQLAlchemy models",
                execution_time_ms=execution_time,
                checksum="sqlalchemy_models"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create security tables: {e}")
            raise
    
    async def run_sql_migration(self, migration_file: str):
        """Run a SQL migration file"""
        try:
            if not os.path.exists(migration_file):
                logger.error(f"Migration file not found: {migration_file}")
                return False
            
            with open(migration_file, 'r') as f:
                sql_content = f.read()
            
            start_time = datetime.now()
            
            async with self.engine.begin() as conn:
                # Split SQL content by statements (simple approach)
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        await conn.execute(text(statement))
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Extract version from filename
            version = os.path.basename(migration_file).split('_')[0]
            description = os.path.basename(migration_file).replace('.sql', '').replace('_', ' ')
            
            await self.record_migration(
                version=version,
                description=description,
                execution_time_ms=execution_time,
                checksum="manual_sql"
            )
            
            logger.info(f"Applied migration: {migration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run migration {migration_file}: {e}")
            raise
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": table_name})
                
                return result.scalar()
                
        except Exception as e:
            logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            async with self.engine.begin() as conn:
                # Get column information
                result = await conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                
                columns = []
                for row in result.fetchall():
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3]
                    })
                
                # Get index information
                index_result = await conn.execute(text("""
                    SELECT indexname, indexdef
                    FROM pg_indexes 
                    WHERE tablename = :table_name
                """), {"table_name": table_name})
                
                indexes = []
                for row in index_result.fetchall():
                    indexes.append({
                        "name": row[0],
                        "definition": row[1]
                    })
                
                return {
                    "columns": columns,
                    "indexes": indexes
                }
                
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {"columns": [], "indexes": []}
    
    async def run_all_migrations(self):
        """Run all pending migrations"""
        try:
            logger.info("Starting database migrations...")
            
            # Initialize migrations table
            await self.initialize_migrations_table()
            
            # Get applied migrations
            applied_migrations = await self.get_applied_migrations()
            logger.info(f"Found {len(applied_migrations)} applied migrations")
            
            # Check if security tables need to be created
            security_tables = [
                "encrypted_api_keys",
                "security_audit_logs", 
                "enhanced_user_sessions",
                "revoked_tokens",
                "api_key_usage_analytics",
                "security_configurations"
            ]
            
            tables_exist = []
            for table in security_tables:
                exists = await self.check_table_exists(table)
                tables_exist.append(exists)
                logger.info(f"Table {table}: {'exists' if exists else 'missing'}")
            
            # If any security tables are missing, create them
            if not all(tables_exist):
                logger.info("Creating missing security tables...")
                await self.create_security_tables()
                logger.info("Security tables created successfully")
            else:
                logger.info("All security tables already exist")
            
            # Run any additional SQL migrations
            migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
            if os.path.exists(migrations_dir):
                sql_files = [f for f in os.listdir(migrations_dir) if f.endswith('.sql')]
                sql_files.sort()
                
                for sql_file in sql_files:
                    version = sql_file.split('_')[0]
                    if version not in applied_migrations:
                        migration_path = os.path.join(migrations_dir, sql_file)
                        await self.run_sql_migration(migration_path)
            
            logger.info("Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def rollback_migration(self, version: str):
        """Rollback a specific migration (basic implementation)"""
        try:
            logger.warning(f"Rolling back migration {version}")
            
            # This is a basic implementation - in production you'd want
            # proper rollback scripts for each migration
            async with self.engine.begin() as conn:
                await conn.execute(text(f"""
                    DELETE FROM {self.migrations_table} 
                    WHERE version = :version
                """), {"version": version})
            
            logger.info(f"Rolled back migration {version}")
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            raise
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get the current migration status"""
        try:
            applied_migrations = await self.get_applied_migrations()
            
            # Check table status
            security_tables = [
                "encrypted_api_keys",
                "security_audit_logs", 
                "enhanced_user_sessions",
                "revoked_tokens",
                "api_key_usage_analytics",
                "security_configurations"
            ]
            
            table_status = {}
            for table in security_tables:
                exists = await self.check_table_exists(table)
                table_status[table] = {
                    "exists": exists,
                    "info": await self.get_table_info(table) if exists else None
                }
            
            return {
                "applied_migrations": applied_migrations,
                "table_status": table_status,
                "migrations_table_exists": await self.check_table_exists(self.migrations_table)
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {}

async def main():
    """Main function to run migrations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        runner = MigrationRunner()
        
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "status":
                status = await runner.get_migration_status()
                print("\n=== Migration Status ===")
                print(f"Applied migrations: {len(status.get('applied_migrations', []))}")
                for migration in status.get('applied_migrations', []):
                    print(f"  - {migration}")
                
                print(f"\nTable status:")
                for table, info in status.get('table_status', {}).items():
                    print(f"  - {table}: {'✓' if info['exists'] else '✗'}")
            
            elif command == "rollback" and len(sys.argv) > 2:
                version = sys.argv[2]
                await runner.rollback_migration(version)
            
            else:
                print("Usage: python migration_runner.py [status|rollback <version>]")
                return
        else:
            # Run all migrations
            await runner.run_all_migrations()
            print("✓ Database migrations completed successfully")
    
    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        print(f"✗ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())