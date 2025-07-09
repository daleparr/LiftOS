"""
KSE Memory SDK Workflow Service
"""

from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import time
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from ..core.interfaces import WorkflowInterface

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class Workflow:
    """Workflow definition and execution state."""
    id: str
    name: str
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


class WorkflowService(WorkflowInterface):
    """
    Workflow service for orchestrating complex KSE Memory operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize workflow service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.task_registry: Dict[str, Callable] = {}
        
        # Configuration - handle both dict and dataclass config
        if hasattr(config, 'get'):
            self.max_concurrent_workflows = config.get('max_concurrent_workflows', 10)
            self.default_timeout = config.get('default_timeout_seconds', 3600)
        else:
            self.max_concurrent_workflows = getattr(config, 'max_concurrent_workflows', 10)
            self.default_timeout = getattr(config, 'default_timeout_seconds', 3600)
        
    async def initialize(self) -> bool:
        """Initialize workflow service."""
        try:
            # Register built-in tasks
            await self._register_builtin_tasks()
            
            logger.info("Workflow service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize workflow service: {e}")
            return False
    
    async def create_workflow(self, name: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            metadata: Additional metadata
            
        Returns:
            Workflow ID
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            workflow = Workflow(
                id=workflow_id,
                name=name,
                metadata=metadata or {}
            )
            
            self.workflows[workflow_id] = workflow
            
            logger.info(f"Created workflow: {workflow_id} ({name})")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def add_task(self, workflow_id: str, task_name: str, 
                      function_name: str, parameters: Optional[Dict[str, Any]] = None,
                      dependencies: Optional[List[str]] = None,
                      max_retries: int = 3, 
                      timeout_seconds: Optional[int] = None) -> str:
        """
        Add a task to a workflow.
        
        Args:
            workflow_id: Workflow ID
            task_name: Task name
            function_name: Function to execute
            parameters: Task parameters
            dependencies: Task dependencies
            max_retries: Maximum retry attempts
            timeout_seconds: Task timeout
            
        Returns:
            Task ID
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.PENDING:
                raise ValueError(f"Cannot add tasks to {workflow.status.value} workflow")
            
            # Get function from registry
            if function_name not in self.task_registry:
                raise ValueError(f"Function not registered: {function_name}")
            
            function = self.task_registry[function_name]
            task_id = str(uuid.uuid4())
            
            task = WorkflowTask(
                id=task_id,
                name=task_name,
                function=function,
                dependencies=dependencies or [],
                parameters=parameters or {},
                max_retries=max_retries,
                timeout_seconds=timeout_seconds
            )
            
            workflow.tasks[task_id] = task
            
            logger.info(f"Added task {task_name} to workflow {workflow_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if execution started successfully
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.PENDING:
                raise ValueError(f"Workflow is {workflow.status.value}, cannot execute")
            
            # Check concurrent workflow limit
            if len(self.running_workflows) >= self.max_concurrent_workflows:
                raise ValueError("Maximum concurrent workflows reached")
            
            # Start workflow execution
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = time.time()
            
            # Create execution task
            execution_task = asyncio.create_task(self._execute_workflow_tasks(workflow))
            self.running_workflows[workflow_id] = execution_task
            
            logger.info(f"Started workflow execution: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            if workflow_id in self.workflows:
                self.workflows[workflow_id].status = WorkflowStatus.FAILED
                self.workflows[workflow_id].error = str(e)
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow status.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status information
        """
        try:
            if workflow_id not in self.workflows:
                return None
            
            workflow = self.workflows[workflow_id]
            
            # Calculate progress
            total_tasks = len(workflow.tasks)
            completed_tasks = sum(
                1 for task in workflow.tasks.values()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            )
            failed_tasks = sum(
                1 for task in workflow.tasks.values()
                if task.status == TaskStatus.FAILED
            )
            
            progress = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            # Calculate duration
            duration = None
            if workflow.started_at:
                end_time = workflow.completed_at or time.time()
                duration = end_time - workflow.started_at
            
            return {
                'id': workflow.id,
                'name': workflow.name,
                'status': workflow.status.value,
                'progress': progress,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'duration': duration,
                'created_at': workflow.created_at,
                'started_at': workflow.started_at,
                'completed_at': workflow.completed_at,
                'error': workflow.error,
                'metadata': workflow.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if cancelled successfully
        """
        try:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status != WorkflowStatus.RUNNING:
                return False
            
            # Cancel execution task
            if workflow_id in self.running_workflows:
                execution_task = self.running_workflows[workflow_id]
                execution_task.cancel()
                del self.running_workflows[workflow_id]
            
            # Update workflow status
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = time.time()
            
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False
    
    async def register_task_function(self, name: str, function: Callable) -> bool:
        """
        Register a task function.
        
        Args:
            name: Function name
            function: Function to register
            
        Returns:
            True if registered successfully
        """
        try:
            self.task_registry[name] = function
            logger.info(f"Registered task function: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register task function: {e}")
            return False
    
    async def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow execution results.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow results
        """
        try:
            if workflow_id not in self.workflows:
                return None
            
            workflow = self.workflows[workflow_id]
            return workflow.results
            
        except Exception as e:
            logger.error(f"Failed to get workflow results: {e}")
            return None
    
    async def list_workflows(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List workflows.
        
        Args:
            status_filter: Filter by status
            
        Returns:
            List of workflow summaries
        """
        try:
            workflows = []
            
            for workflow in self.workflows.values():
                if status_filter and workflow.status.value != status_filter:
                    continue
                
                workflows.append({
                    'id': workflow.id,
                    'name': workflow.name,
                    'status': workflow.status.value,
                    'task_count': len(workflow.tasks),
                    'created_at': workflow.created_at,
                    'started_at': workflow.started_at,
                    'completed_at': workflow.completed_at
                })
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return []
    
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Get workflow execution history.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            List of workflow execution history entries
        """
        try:
            if workflow_id not in self.workflows:
                return []
            
            workflow = self.workflows[workflow_id]
            history = []
            
            # Add workflow creation event
            history.append({
                'timestamp': workflow.created_at,
                'event_type': 'workflow_created',
                'event_data': {
                    'workflow_id': workflow.id,
                    'workflow_name': workflow.name,
                    'task_count': len(workflow.tasks)
                }
            })
            
            # Add workflow start event if started
            if workflow.started_at:
                history.append({
                    'timestamp': workflow.started_at,
                    'event_type': 'workflow_started',
                    'event_data': {
                        'workflow_id': workflow.id,
                        'status': workflow.status.value
                    }
                })
            
            # Add task execution events
            for task in workflow.tasks.values():
                if task.started_at:
                    history.append({
                        'timestamp': task.started_at,
                        'event_type': 'task_started',
                        'event_data': {
                            'workflow_id': workflow.id,
                            'task_id': task.id,
                            'task_name': task.name,
                            'task_type': task.task_type
                        }
                    })
                
                if task.completed_at:
                    history.append({
                        'timestamp': task.completed_at,
                        'event_type': 'task_completed',
                        'event_data': {
                            'workflow_id': workflow.id,
                            'task_id': task.id,
                            'task_name': task.name,
                            'status': task.status.value,
                            'retry_count': task.retry_count,
                            'error': task.error
                        }
                    })
            
            # Add workflow completion event if completed
            if workflow.completed_at:
                history.append({
                    'timestamp': workflow.completed_at,
                    'event_type': 'workflow_completed',
                    'event_data': {
                        'workflow_id': workflow.id,
                        'status': workflow.status.value,
                        'duration': workflow.completed_at - (workflow.started_at or workflow.created_at),
                        'error': workflow.error
                    }
                })
            
            # Sort history by timestamp
            history.sort(key=lambda x: x['timestamp'])
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get workflow history: {e}")
            return []
    
    async def _execute_workflow_tasks(self, workflow: Workflow) -> None:
        """Execute workflow tasks."""
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow)
            
            # Execute tasks in dependency order
            executed_tasks = set()
            
            while len(executed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task_id, task in workflow.tasks.items():
                    if (task_id not in executed_tasks and 
                        task.status == TaskStatus.PENDING and
                        all(dep in executed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check for circular dependencies or failed dependencies
                    remaining_tasks = [
                        task for task_id, task in workflow.tasks.items()
                        if task_id not in executed_tasks
                    ]
                    
                    if any(task.status == TaskStatus.FAILED for task in remaining_tasks):
                        # Some dependencies failed
                        for task in remaining_tasks:
                            if task.status == TaskStatus.PENDING:
                                task.status = TaskStatus.SKIPPED
                        break
                    else:
                        # Circular dependency
                        raise ValueError("Circular dependency detected in workflow")
                
                # Execute ready tasks concurrently
                if ready_tasks:
                    tasks = [self._execute_task(task, workflow) for task in ready_tasks]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Mark tasks as executed
                    for task in ready_tasks:
                        executed_tasks.add(task.id)
            
            # Update workflow status
            failed_tasks = [
                task for task in workflow.tasks.values()
                if task.status == TaskStatus.FAILED
            ]
            
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = f"{len(failed_tasks)} tasks failed"
            else:
                workflow.status = WorkflowStatus.COMPLETED
            
            workflow.completed_at = time.time()
            
            # Collect results
            workflow.results = {
                task.name: task.result for task in workflow.tasks.values()
                if task.result is not None
            }
            
        except asyncio.CancelledError:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = time.time()
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = time.time()
            logger.error(f"Workflow execution failed: {e}")
        finally:
            # Clean up running workflow
            if workflow.id in self.running_workflows:
                del self.running_workflows[workflow.id]
    
    async def _execute_task(self, task: WorkflowTask, workflow: Workflow) -> None:
        """Execute a single task."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            # Execute with timeout
            if task.timeout_seconds:
                task.result = await asyncio.wait_for(
                    task.function(**task.parameters),
                    timeout=task.timeout_seconds
                )
            else:
                task.result = await task.function(**task.parameters)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = "Task timeout"
            task.completed_at = time.time()
        except Exception as e:
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Retry task
                task.status = TaskStatus.PENDING
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(task, workflow)
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
    
    def _build_dependency_graph(self, workflow: Workflow) -> Dict[str, List[str]]:
        """Build dependency graph for workflow tasks."""
        graph = {}
        for task_id, task in workflow.tasks.items():
            graph[task_id] = task.dependencies
        return graph
    
    async def _register_builtin_tasks(self) -> None:
        """Register built-in task functions."""
        
        async def sleep_task(duration: float = 1.0) -> str:
            """Sleep for specified duration."""
            await asyncio.sleep(duration)
            return f"Slept for {duration} seconds"
        
        async def log_task(message: str = "Task executed") -> str:
            """Log a message."""
            logger.info(f"Workflow task: {message}")
            return message
        
        # Register built-in tasks
        await self.register_task_function("sleep", sleep_task)
        await self.register_task_function("log", log_task)