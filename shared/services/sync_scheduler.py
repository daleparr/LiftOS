"""
Background Sync Scheduler
Handles automated data synchronization for platform connections
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from shared.database.user_platform_models import UserPlatformConnection, DataSyncLog
from shared.services.platform_connection_service import get_platform_connection_service
from shared.connectors.connector_factory import get_connector_manager
from shared.utils.logging import setup_logging
from shared.models.marketing import DataSource

logger = setup_logging("sync_scheduler")

class SyncStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SyncJob:
    """Represents a sync job"""
    job_id: str
    connection_id: str
    user_id: str
    org_id: str
    platform: DataSource
    sync_type: str
    sync_config: Dict[str, Any]
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: SyncStatus = SyncStatus.PENDING
    error_message: Optional[str] = None
    records_synced: int = 0
    retry_count: int = 0
    max_retries: int = 3

class SyncScheduler:
    """Background scheduler for platform data synchronization"""
    
    def __init__(self):
        self.running = False
        self.sync_jobs: Dict[str, SyncJob] = {}
        self.active_syncs: Dict[str, asyncio.Task] = {}
        self.platform_service = None
        self.connector_manager = get_connector_manager()
        self.max_concurrent_syncs = 5
        self.sync_interval = 300  # 5 minutes
        
    async def start(self):
        """Start the sync scheduler"""
        if self.running:
            logger.warning("Sync scheduler is already running")
            return
        
        self.running = True
        self.platform_service = get_platform_connection_service()
        
        logger.info("Starting sync scheduler")
        
        # Start the main scheduler loop
        asyncio.create_task(self._scheduler_loop())
        
        # Start the job processor
        asyncio.create_task(self._job_processor())
        
    async def stop(self):
        """Stop the sync scheduler"""
        if not self.running:
            return
        
        logger.info("Stopping sync scheduler")
        self.running = False
        
        # Cancel all active sync tasks
        for job_id, task in self.active_syncs.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled sync job {job_id}")
        
        # Wait for tasks to complete
        if self.active_syncs:
            await asyncio.gather(*self.active_syncs.values(), return_exceptions=True)
        
        self.active_syncs.clear()
        
    async def schedule_sync(self, connection_id: str, user_id: str, org_id: str,
                          sync_type: str = "incremental", 
                          sync_config: Optional[Dict[str, Any]] = None,
                          delay_seconds: int = 0) -> str:
        """Schedule a sync job"""
        try:
            # Get connection details
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            connection = next((c for c in connections if c.id == connection_id), None)
            
            if not connection:
                raise ValueError(f"Connection {connection_id} not found")
            
            # Create sync job
            job_id = str(uuid.uuid4())
            scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            
            sync_job = SyncJob(
                job_id=job_id,
                connection_id=connection_id,
                user_id=user_id,
                org_id=org_id,
                platform=DataSource(connection.platform),
                sync_type=sync_type,
                sync_config=sync_config or {},
                scheduled_at=scheduled_at
            )
            
            self.sync_jobs[job_id] = sync_job
            logger.info(f"Scheduled sync job {job_id} for connection {connection_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to schedule sync job: {str(e)}")
            raise
    
    async def cancel_sync(self, job_id: str) -> bool:
        """Cancel a sync job"""
        if job_id not in self.sync_jobs:
            return False
        
        sync_job = self.sync_jobs[job_id]
        
        if sync_job.status == SyncStatus.RUNNING and job_id in self.active_syncs:
            # Cancel the running task
            task = self.active_syncs[job_id]
            task.cancel()
            sync_job.status = SyncStatus.CANCELLED
            logger.info(f"Cancelled running sync job {job_id}")
        elif sync_job.status == SyncStatus.PENDING:
            # Mark as cancelled
            sync_job.status = SyncStatus.CANCELLED
            logger.info(f"Cancelled pending sync job {job_id}")
        else:
            return False
        
        return True
    
    async def get_sync_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get sync job status"""
        if job_id not in self.sync_jobs:
            return None
        
        sync_job = self.sync_jobs[job_id]
        
        return {
            "job_id": job_id,
            "connection_id": sync_job.connection_id,
            "platform": sync_job.platform.value,
            "sync_type": sync_job.sync_type,
            "status": sync_job.status.value,
            "scheduled_at": sync_job.scheduled_at.isoformat(),
            "started_at": sync_job.started_at.isoformat() if sync_job.started_at else None,
            "completed_at": sync_job.completed_at.isoformat() if sync_job.completed_at else None,
            "records_synced": sync_job.records_synced,
            "error_message": sync_job.error_message,
            "retry_count": sync_job.retry_count
        }
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                await self._schedule_automatic_syncs()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _schedule_automatic_syncs(self):
        """Schedule automatic syncs based on connection settings"""
        try:
            if not self.platform_service:
                return
            
            # Get all active connections that need syncing
            # This would typically query the database for connections with auto_sync_enabled
            # For now, we'll implement a basic version
            
            # Get connections that haven't been synced recently
            cutoff_time = datetime.utcnow() - timedelta(hours=1)  # Sync every hour
            
            # This is a simplified implementation
            # In production, you'd query the database for connections needing sync
            
        except Exception as e:
            logger.error(f"Error scheduling automatic syncs: {str(e)}")
    
    async def _job_processor(self):
        """Process pending sync jobs"""
        while self.running:
            try:
                # Find jobs ready to run
                current_time = datetime.utcnow()
                ready_jobs = [
                    job for job in self.sync_jobs.values()
                    if (job.status == SyncStatus.PENDING and 
                        job.scheduled_at <= current_time and
                        len(self.active_syncs) < self.max_concurrent_syncs)
                ]
                
                # Start ready jobs
                for job in ready_jobs:
                    if job.job_id not in self.active_syncs:
                        task = asyncio.create_task(self._execute_sync_job(job))
                        self.active_syncs[job.job_id] = task
                
                # Clean up completed tasks
                completed_jobs = [
                    job_id for job_id, task in self.active_syncs.items()
                    if task.done()
                ]
                
                for job_id in completed_jobs:
                    del self.active_syncs[job_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in job processor: {str(e)}")
                await asyncio.sleep(30)
    
    async def _execute_sync_job(self, sync_job: SyncJob):
        """Execute a sync job"""
        try:
            sync_job.status = SyncStatus.RUNNING
            sync_job.started_at = datetime.utcnow()
            
            logger.info(f"Starting sync job {sync_job.job_id} for platform {sync_job.platform.value}")
            
            # Get connection details
            connections = await self.platform_service.get_user_connections(
                sync_job.user_id, sync_job.org_id
            )
            connection = next((c for c in connections if c.id == sync_job.connection_id), None)
            
            if not connection:
                raise Exception(f"Connection {sync_job.connection_id} not found")
            
            # Get connector
            connector = await self.connector_manager.get_connector(
                sync_job.connection_id,
                sync_job.platform,
                connection.credentials
            )
            
            # Determine date range for sync
            end_date = datetime.utcnow().date()
            
            if sync_job.sync_type == "incremental":
                # Get last sync date or default to 7 days ago
                start_date = connection.last_sync_at.date() if connection.last_sync_at else (end_date - timedelta(days=7))
            else:  # full sync
                days_back = sync_job.sync_config.get("days_back", 30)
                start_date = end_date - timedelta(days=days_back)
            
            # Extract data
            data = await connector.extract_data(start_date, end_date)
            
            # Process and store data
            records_synced = await self._process_sync_data(
                data, sync_job, connection
            )
            
            # Update sync job
            sync_job.status = SyncStatus.COMPLETED
            sync_job.completed_at = datetime.utcnow()
            sync_job.records_synced = records_synced
            
            # Log successful sync
            await self._log_sync_result(sync_job, True)
            
            logger.info(f"Completed sync job {sync_job.job_id}, synced {records_synced} records")
            
        except asyncio.CancelledError:
            sync_job.status = SyncStatus.CANCELLED
            logger.info(f"Sync job {sync_job.job_id} was cancelled")
            
        except Exception as e:
            sync_job.status = SyncStatus.FAILED
            sync_job.error_message = str(e)
            sync_job.completed_at = datetime.utcnow()
            
            logger.error(f"Sync job {sync_job.job_id} failed: {str(e)}")
            
            # Log failed sync
            await self._log_sync_result(sync_job, False)
            
            # Schedule retry if within retry limit
            if sync_job.retry_count < sync_job.max_retries:
                await self._schedule_retry(sync_job)
    
    async def _process_sync_data(self, data: List[Dict[str, Any]], 
                               sync_job: SyncJob, connection: Any) -> int:
        """Process and store synced data"""
        try:
            # Transform data to causal format if needed
            from shared.utils.causal_transforms import CausalDataTransformer
            
            transformer = CausalDataTransformer()
            processed_data = []
            
            for record in data:
                try:
                    # Transform to causal format
                    causal_data = await transformer.transform_to_causal_format(
                        raw_data=record,
                        data_source=sync_job.platform,
                        org_id=sync_job.org_id
                    )
                    processed_data.append(causal_data)
                    
                except Exception as e:
                    logger.warning(f"Failed to transform record: {str(e)}")
                    continue
            
            # Send to memory service
            if processed_data:
                await self._send_to_memory_service(processed_data, sync_job)
            
            return len(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing sync data: {str(e)}")
            raise
    
    async def _send_to_memory_service(self, data: List[Any], sync_job: SyncJob):
        """Send processed data to memory service"""
        try:
            import httpx
            
            memory_service_url = "http://localhost:8003"  # Should be configurable
            
            payload = {
                "data": [item.model_dump() if hasattr(item, 'model_dump') else item for item in data],
                "source": sync_job.platform.value,
                "org_id": sync_job.org_id,
                "sync_job_id": sync_job.job_id
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{memory_service_url}/api/v1/data/ingest",
                    json=payload,
                    headers={
                        "X-User-ID": sync_job.user_id,
                        "X-Org-ID": sync_job.org_id
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Memory service returned {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to send data to memory service: {str(e)}")
            raise
    
    async def _log_sync_result(self, sync_job: SyncJob, success: bool):
        """Log sync result to database"""
        try:
            # This would typically create a DataSyncLog record
            # For now, we'll just log it
            
            log_data = {
                "connection_id": sync_job.connection_id,
                "sync_type": sync_job.sync_type,
                "status": "completed" if success else "failed",
                "records_synced": sync_job.records_synced,
                "started_at": sync_job.started_at,
                "completed_at": sync_job.completed_at,
                "error_message": sync_job.error_message,
                "sync_config": sync_job.sync_config
            }
            
            logger.info(f"Sync result logged: {log_data}")
            
        except Exception as e:
            logger.error(f"Failed to log sync result: {str(e)}")
    
    async def _schedule_retry(self, sync_job: SyncJob):
        """Schedule a retry for a failed sync job"""
        try:
            sync_job.retry_count += 1
            
            # Exponential backoff: 2^retry_count minutes
            delay_minutes = 2 ** sync_job.retry_count
            retry_time = datetime.utcnow() + timedelta(minutes=delay_minutes)
            
            # Create new job for retry
            retry_job = SyncJob(
                job_id=str(uuid.uuid4()),
                connection_id=sync_job.connection_id,
                user_id=sync_job.user_id,
                org_id=sync_job.org_id,
                platform=sync_job.platform,
                sync_type=sync_job.sync_type,
                sync_config=sync_job.sync_config,
                scheduled_at=retry_time,
                retry_count=sync_job.retry_count
            )
            
            self.sync_jobs[retry_job.job_id] = retry_job
            
            logger.info(f"Scheduled retry {sync_job.retry_count} for sync job {sync_job.job_id} "
                       f"at {retry_time.isoformat()}")
            
        except Exception as e:
            logger.error(f"Failed to schedule retry: {str(e)}")

# Global sync scheduler instance
_sync_scheduler = None

async def get_sync_scheduler() -> SyncScheduler:
    """Get the global sync scheduler instance"""
    global _sync_scheduler
    
    if _sync_scheduler is None:
        _sync_scheduler = SyncScheduler()
        await _sync_scheduler.start()
    
    return _sync_scheduler

async def shutdown_sync_scheduler():
    """Shutdown the global sync scheduler"""
    global _sync_scheduler
    
    if _sync_scheduler:
        await _sync_scheduler.stop()
        _sync_scheduler = None