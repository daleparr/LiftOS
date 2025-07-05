"""
Job Scheduler for Data Ingestion Service
Handles automated sync jobs and scheduling
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

from shared.models.marketing import DataSource

logger = logging.getLogger(__name__)

class ScheduleFrequency(Enum):
    """Supported schedule frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class ScheduledJob:
    """Represents a scheduled sync job"""
    job_id: str
    org_id: str
    platform: DataSource
    frequency: ScheduleFrequency
    next_run: datetime
    last_run: Optional[datetime] = None
    is_active: bool = True
    sync_type: str = "full"
    date_range_days: int = 30

class JobScheduler:
    """Manages scheduled sync jobs"""
    
    def __init__(self):
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the job scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job scheduler started")
    
    async def stop(self):
        """Stop the job scheduler"""
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Job scheduler stopped")
    
    async def add_scheduled_job(
        self,
        org_id: str,
        platform: DataSource,
        frequency: ScheduleFrequency,
        sync_type: str = "full",
        date_range_days: int = 30
    ) -> str:
        """Add a new scheduled job"""
        job_id = str(uuid.uuid4())
        next_run = self._calculate_next_run(frequency)
        
        scheduled_job = ScheduledJob(
            job_id=job_id,
            org_id=org_id,
            platform=platform,
            frequency=frequency,
            next_run=next_run,
            sync_type=sync_type,
            date_range_days=date_range_days
        )
        
        self.scheduled_jobs[job_id] = scheduled_job
        logger.info(f"Added scheduled job {job_id} for {platform.value} (org: {org_id})")
        
        return job_id
    
    async def remove_scheduled_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        if job_id in self.scheduled_jobs:
            del self.scheduled_jobs[job_id]
            logger.info(f"Removed scheduled job {job_id}")
            return True
        return False
    
    async def update_scheduled_job(
        self,
        job_id: str,
        frequency: Optional[ScheduleFrequency] = None,
        is_active: Optional[bool] = None,
        sync_type: Optional[str] = None,
        date_range_days: Optional[int] = None
    ) -> bool:
        """Update a scheduled job"""
        if job_id not in self.scheduled_jobs:
            return False
        
        job = self.scheduled_jobs[job_id]
        
        if frequency is not None:
            job.frequency = frequency
            job.next_run = self._calculate_next_run(frequency)
        
        if is_active is not None:
            job.is_active = is_active
        
        if sync_type is not None:
            job.sync_type = sync_type
        
        if date_range_days is not None:
            job.date_range_days = date_range_days
        
        logger.info(f"Updated scheduled job {job_id}")
        return True
    
    async def get_scheduled_jobs(self, org_id: Optional[str] = None) -> List[ScheduledJob]:
        """Get scheduled jobs, optionally filtered by org_id"""
        jobs = list(self.scheduled_jobs.values())
        
        if org_id:
            jobs = [job for job in jobs if job.org_id == org_id]
        
        return jobs
    
    async def get_scheduled_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a specific scheduled job"""
        return self.scheduled_jobs.get(job_id)
    
    def _calculate_next_run(self, frequency: ScheduleFrequency) -> datetime:
        """Calculate next run time based on frequency"""
        now = datetime.utcnow()
        
        if frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == ScheduleFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ScheduleFrequency.MONTHLY:
            return now + timedelta(days=30)
        else:
            return now + timedelta(days=1)  # Default to daily
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _check_and_run_jobs(self):
        """Check for jobs that need to run and execute them"""
        now = datetime.utcnow()
        
        for job in self.scheduled_jobs.values():
            if not job.is_active:
                continue
            
            if job.next_run <= now:
                await self._execute_scheduled_job(job)
    
    async def _execute_scheduled_job(self, job: ScheduledJob):
        """Execute a scheduled job"""
        try:
            logger.info(f"Executing scheduled job {job.job_id} for {job.platform.value}")
            
            # Import here to avoid circular imports
            from app import create_sync_job
            
            # Calculate date range
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=job.date_range_days)
            
            # Create sync job request
