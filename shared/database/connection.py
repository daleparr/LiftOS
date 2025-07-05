"""
Database connection management for Lift OS Core
"""

import os
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()
metadata = MetaData()

class DatabaseManager:
    """Manages database connections for PostgreSQL and Redis"""
    
    def __init__(self):
        self.pg_engine = None
        self.pg_session_factory = None
        self.redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return
            
        try:
            # Database connection (SQLite for local development)
            # Use absolute path to ensure it works from any service directory
            # Go up 3 levels from shared/database/connection.py to get to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_path = os.path.join(project_root, "data", "lift_os_dev.db")
            database_url = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
            self.pg_engine = create_async_engine(
                database_url,
                echo=os.getenv("DEBUG", "false").lower() == "true",
                pool_size=10,
                max_overflow=20
            )
            
            self.pg_session_factory = async_sessionmaker(
                self.pg_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connections
            await self._test_connections()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _test_connections(self):
        """Test database connections"""
        # Test PostgreSQL
        try:
            async with self.pg_engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("PostgreSQL connection test successful")
        except Exception as e:
            logger.warning(f"PostgreSQL connection test failed: {e}")
            # For local development, we'll continue without PostgreSQL
            
        # Test Redis
        try:
            await self.redis_client.ping()
            logger.info("Redis connection test successful")
        except Exception as e:
            logger.warning(f"Redis connection test failed: {e}")
            # For local development, we'll continue without Redis
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session"""
        if not self._initialized:
            await self.initialize()
            
        if not self.pg_session_factory:
            raise RuntimeError("Database not initialized")
            
        async with self.pg_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self._initialized:
            await self.initialize()
            
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
            
        return self.redis_client
    
    async def close(self):
        """Close all database connections"""
        if self.pg_engine:
            await self.pg_engine.dispose()
            
        if self.redis_client:
            await self.redis_client.close()
            
        self._initialized = False
        logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

async def get_database() -> DatabaseManager:
    """Get the global database manager"""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager

# Convenience functions
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session (convenience function)"""
    async with db_manager.get_session() as session:
        yield session

async def get_redis() -> redis.Redis:
    """Get Redis client (convenience function)"""
    return await db_manager.get_redis()