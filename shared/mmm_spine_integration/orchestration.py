"""
Ultra-Fast Orchestration Framework - Adapted from MMM Spine
Provides sub-2-second execution through aggressive optimization and caching
"""

import asyncio
import time
import threading
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import concurrent.futures
from contextlib import asynccontextmanager
import weakref

from .observability import ObservabilityManager, TraceLevel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class WorkflowTask:
    """Represents a workflow task with execution metadata."""
    task_id: str
    workflow_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time_ms: Optional[float] = None


@dataclass
class Workflow:
    """Represents a workflow with tasks and execution state."""
    workflow_id: str
    name: str
    org_id: str
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionCache:
    """Cache entry for function execution results."""
    result: Any
    timestamp: float
    execution_time_ms: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class UltraFastOrchestrator:
    """
    Ultra-fast orchestrator with aggressive optimization and caching.
    Adapted from MMM Spine ultra_fast_orchestrator.py
    """
    
    def __init__(self, max_workers: int = 10, cache_size: int = 10000,
                 cache_ttl: int = 3600, observability: Optional[ObservabilityManager] = None):
        """
        Initialize ultra-fast orchestrator.
        
        Args:
            max_workers: Maximum number of worker threads
            cache_size: Maximum number of cached results
            cache_ttl: Cache time-to-live in seconds
            observability: Optional observability manager
        """
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.observability = observability
        
        # Execution infrastructure
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Workflow storage
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}
        
        # Execution cache with LRU eviction
        self.execution_cache: Dict[str, ExecutionCache] = {}
        self.cache_access_order: deque = deque(maxlen=cache_size)
        
        # Performance optimization
        self.function_registry: Dict[str, Callable] = {}
        self.compiled_workflows: Dict[str, Callable] = {}
        
        # Performance tracking
        self.total_executions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_execution_time_ms = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized UltraFastOrchestrator with {max_workers} workers")
    
    def _generate_cache_key(self, function_name: str, args: tuple, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for function execution."""
        # Create deterministic hash of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        return f"{function_name}:{hash(args_str)}"
    
    def _is_cache_entry_valid(self, entry: ExecutionCache) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - entry.timestamp) < self.cache_ttl
    
    def _evict_lru_cache_entry(self):
        """Evict least recently used cache entry."""
        if self.cache_access_order:
            lru_key = self.cache_access_order.popleft()
            if lru_key in self.execution_cache:
                del self.execution_cache[lru_key]
    
    def register_function(self, name: str, function: Callable):
        """
        Register a function for optimized execution.
        
        Args:
            name: Function name
            function: Function to register
        """
        self.function_registry[name] = function
        logger.debug(f"Registered function: {name}")
    
    async def execute_function_cached(self, function_name: str, args: tuple = (),
                                    kwargs: Dict[str, Any] = None) -> Any:
        """
        Execute function with aggressive caching.
        
        Args:
            function_name: Name of the function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        kwargs = kwargs or {}
        cache_key = self._generate_cache_key(function_name, args, kwargs)
        
        # Check cache first
        with self.lock:
            if cache_key in self.execution_cache:
                entry = self.execution_cache[cache_key]
                if self._is_cache_entry_valid(entry):
                    # Cache hit
                    entry.hit_count += 1
                    entry.last_accessed = time.time()
                    
                    # Update access order
                    if cache_key in self.cache_access_order:
                        self.cache_access_order.remove(cache_key)
                    self.cache_access_order.append(cache_key)
                    
                    self.cache_hits += 1
                    
                    if self.observability:
                        self.observability.metrics.increment_counter(
                            "orchestrator.cache_hit",
                            tags={"function": function_name}
                        )
                    
                    logger.debug(f"Cache hit for {function_name}")
                    return entry.result
                else:
                    # Cache entry expired
                    del self.execution_cache[cache_key]
            
            # Cache miss
            self.cache_misses += 1
        
        # Execute function
        start_time = time.time()
        
        try:
            if function_name in self.function_registry:
                function = self.function_registry[function_name]
            else:
                raise ValueError(f"Function not registered: {function_name}")
            
            # Execute with observability
            if self.observability:
                async with self.observability.observe_async_operation(
                    f"orchestrator.execute.{function_name}",
                    tags={"function": function_name},
                    level=TraceLevel.DEBUG
                ):
                    if asyncio.iscoroutinefunction(function):
                        result = await function(*args, **kwargs)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor, lambda: function(*args, **kwargs)
                        )
            else:
                if asyncio.iscoroutinefunction(function):
                    result = await function(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, lambda: function(*args, **kwargs)
                    )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            with self.lock:
                if len(self.execution_cache) >= self.cache_size:
                    self._evict_lru_cache_entry()
                
                cache_entry = ExecutionCache(
                    result=result,
                    timestamp=time.time(),
                    execution_time_ms=execution_time_ms,
                    hit_count=0
                )
                
                self.execution_cache[cache_key] = cache_entry
                self.cache_access_order.append(cache_key)
                
                self.total_executions += 1
                self.total_execution_time_ms += execution_time_ms
            
            if self.observability:
                self.observability.metrics.record_timer(
                    "orchestrator.execution_time",
                    execution_time_ms,
                    tags={"function": function_name}
                )
            
            logger.debug(f"Executed {function_name} in {execution_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            if self.observability:
                self.observability.metrics.increment_counter(
                    "orchestrator.execution_error",
                    tags={"function": function_name, "error": type(e).__name__}
                )
            
            logger.error(f"Function execution failed: {function_name} - {str(e)}")
            raise
    
    def create_workflow(self, name: str, org_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            org_id: Organization ID
            metadata: Optional metadata
            
        Returns:
            Workflow ID
        """
        workflow_id = f"wf_{org_id}_{uuid.uuid4().hex[:8]}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            org_id=org_id,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow: {workflow_id} for org {org_id}")
        return workflow_id
    
    def add_task(self, workflow_id: str, task_name: str, function_name: str,
                args: tuple = (), kwargs: Dict[str, Any] = None,
                dependencies: List[str] = None, priority: TaskPriority = TaskPriority.NORMAL,
                timeout_seconds: Optional[float] = None, max_retries: int = 3) -> str:
        """
        Add a task to a workflow.
        
        Args:
            workflow_id: Workflow ID
            task_name: Task name
            function_name: Name of function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            dependencies: List of task IDs this task depends on
            priority: Task priority
            timeout_seconds: Task timeout
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if function_name not in self.function_registry:
            raise ValueError(f"Function not registered: {function_name}")
        
        task_id = f"task_{workflow_id}_{uuid.uuid4().hex[:8]}"
        
        task = WorkflowTask(
            task_id=task_id,
            workflow_id=workflow_id,
            name=task_name,
            function=self.function_registry[function_name],
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or [],
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        with self.lock:
            workflow = self.workflows[workflow_id]
            workflow.tasks[task_id] = task
            workflow.task_order.append(task_id)
        
        logger.debug(f"Added task {task_name} to workflow {workflow_id}")
        return task_id
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a workflow with ultra-fast optimization.
        
        Args:
            workflow_id: Workflow ID
            context: Optional execution context
            
        Returns:
            Workflow execution results
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        workflow.context.update(context or {})
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = time.time()
        
        if self.observability:
            async with self.observability.observe_async_operation(
                f"orchestrator.workflow.{workflow.name}",
                tags={"workflow_id": workflow_id, "org_id": workflow.org_id},
                level=TraceLevel.INFO
            ) as span:
                return await self._execute_workflow_internal(workflow, span)
        else:
            return await self._execute_workflow_internal(workflow)
    
    async def _execute_workflow_internal(self, workflow: Workflow, span=None) -> Dict[str, Any]:
        """Internal workflow execution with dependency resolution."""
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow)
            
            # Execute tasks in dependency order with parallelization
            task_results = {}
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute (dependencies satisfied)
                ready_tasks = []
                for task_id, task in workflow.tasks.items():
                    if (task_id not in completed_tasks and 
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check for circular dependencies
                    remaining_tasks = [t for t in workflow.tasks.values() 
                                     if t.task_id not in completed_tasks]
                    raise RuntimeError(f"Circular dependency detected in workflow {workflow.workflow_id}")
                
                # Sort by priority
                ready_tasks.sort(key=lambda t: t.priority.value)
                
                # Execute ready tasks in parallel
                if len(ready_tasks) == 1:
                    # Single task - execute directly
                    task = ready_tasks[0]
                    result = await self._execute_task(task, workflow.context)
                    task_results[task.task_id] = result
                    completed_tasks.add(task.task_id)
                else:
                    # Multiple tasks - execute in parallel
                    task_coroutines = [
                        self._execute_task(task, workflow.context) 
                        for task in ready_tasks
                    ]
                    
                    results = await asyncio.gather(*task_coroutines, return_exceptions=True)
                    
                    for task, result in zip(ready_tasks, results):
                        if isinstance(result, Exception):
                            task.status = WorkflowStatus.FAILED
                            task.error = str(result)
                            raise result
                        else:
                            task_results[task.task_id] = result
                            completed_tasks.add(task.task_id)
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = time.time()
            workflow.total_execution_time_ms = (workflow.end_time - workflow.start_time) * 1000
            
            if self.observability:
                self.observability.metrics.record_timer(
                    "orchestrator.workflow_execution_time",
                    workflow.total_execution_time_ms,
                    tags={"workflow_name": workflow.name, "org_id": workflow.org_id}
                )
                
                if span:
                    self.observability.tracer.add_span_tag(
                        span, "workflow.execution_time_ms", workflow.total_execution_time_ms
                    )
                    self.observability.tracer.add_span_tag(
                        span, "workflow.tasks_count", len(workflow.tasks)
                    )
            
            logger.info(f"Workflow {workflow.workflow_id} completed in "
                       f"{workflow.total_execution_time_ms:.2f}ms")
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": workflow.status.value,
                "execution_time_ms": workflow.total_execution_time_ms,
                "task_results": task_results,
                "tasks_executed": len(completed_tasks)
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = time.time()
            
            if self.observability:
                self.observability.metrics.increment_counter(
                    "orchestrator.workflow_error",
                    tags={"workflow_name": workflow.name, "error": type(e).__name__}
                )
            
            logger.error(f"Workflow {workflow.workflow_id} failed: {str(e)}")
            raise
    
    async def _execute_task(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Execute a single task with retry logic."""
        task.start_time = time.time()
        task.status = WorkflowStatus.RUNNING
        
        for attempt in range(task.max_retries + 1):
            try:
                # Merge context into kwargs
                merged_kwargs = {**task.kwargs, **context}
                
                # Execute with timeout
                if task.timeout_seconds:
                    result = await asyncio.wait_for(
                        self.execute_function_cached(
                            task.function.__name__,
                            task.args,
                            merged_kwargs
                        ),
                        timeout=task.timeout_seconds
                    )
                else:
                    result = await self.execute_function_cached(
                        task.function.__name__,
                        task.args,
                        merged_kwargs
                    )
                
                # Task completed successfully
                task.status = WorkflowStatus.COMPLETED
                task.result = result
                task.end_time = time.time()
                task.execution_time_ms = (task.end_time - task.start_time) * 1000
                
                if self.observability:
                    self.observability.metrics.record_timer(
                        "orchestrator.task_execution_time",
                        task.execution_time_ms,
                        tags={"task_name": task.name, "attempt": str(attempt + 1)}
                    )
                
                logger.debug(f"Task {task.name} completed in {task.execution_time_ms:.2f}ms")
                return result
                
            except Exception as e:
                task.retry_count = attempt + 1
                
                if attempt < task.max_retries:
                    # Retry with exponential backoff
                    backoff_delay = 0.1 * (2 ** attempt)
                    await asyncio.sleep(backoff_delay)
                    
                    logger.warning(f"Task {task.name} failed (attempt {attempt + 1}), "
                                 f"retrying in {backoff_delay}s: {str(e)}")
                else:
                    # Max retries exceeded
                    task.status = WorkflowStatus.FAILED
                    task.error = str(e)
                    task.end_time = time.time()
                    
                    if self.observability:
                        self.observability.metrics.increment_counter(
                            "orchestrator.task_error",
                            tags={"task_name": task.name, "error": type(e).__name__}
                        )
                    
                    logger.error(f"Task {task.name} failed after {task.max_retries} retries: {str(e)}")
                    raise
    
    def _build_dependency_graph(self, workflow: Workflow) -> Dict[str, List[str]]:
        """Build dependency graph for workflow tasks."""
        graph = {}
        
        for task_id, task in workflow.tasks.items():
            graph[task_id] = task.dependencies.copy()
        
        return graph
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration performance statistics."""
        with self.lock:
            cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0)
            
            avg_execution_time = (self.total_execution_time_ms / self.total_executions 
                                if self.total_executions > 0 else 0)
            
            return {
                "total_executions": self.total_executions,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.execution_cache),
                "max_cache_size": self.cache_size,
                "average_execution_time_ms": avg_execution_time,
                "total_workflows": len(self.workflows),
                "active_workflows": len(self.active_workflows),
                "registered_functions": len(self.function_registry)
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        
        task_statuses = {}
        for task_id, task in workflow.tasks.items():
            task_statuses[task_id] = {
                "name": task.name,
                "status": task.status.value,
                "execution_time_ms": task.execution_time_ms,
                "retry_count": task.retry_count,
                "error": task.error
            }
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "total_execution_time_ms": workflow.total_execution_time_ms,
            "tasks": task_statuses,
            "metadata": workflow.metadata
        }


class WorkflowOrchestrator:
    """
    High-level workflow orchestrator with template support.
    Adapted from MMM Spine workflow_orchestrator.py
    """
    
    def __init__(self, ultra_fast_orchestrator: UltraFastOrchestrator):
        """
        Initialize workflow orchestrator.
        
        Args:
            ultra_fast_orchestrator: Ultra-fast orchestrator instance
        """
        self.orchestrator = ultra_fast_orchestrator
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized WorkflowOrchestrator")
    
    def register_workflow_template(self, template_name: str, template: Dict[str, Any]):
        """
        Register a workflow template for reuse.
        
        Args:
            template_name: Template name
            template: Template definition
        """
        self.workflow_templates[template_name] = template
        logger.info(f"Registered workflow template: {template_name}")
    
    async def create_workflow_from_template(self, template_name: str, org_id: str,
                                          parameters: Dict[str, Any] = None) -> str:
        """
        Create workflow from template.
        
        Args:
            template_name: Template name
            org_id: Organization ID
            parameters: Template parameters
            
        Returns:
            Workflow ID
        """
        if template_name not in self.workflow_templates:
            raise ValueError(f"Workflow template not found: {template_name}")
        
        template = self.workflow_templates[template_name]
        parameters = parameters or {}
        
        # Create workflow
        workflow_id = self.orchestrator.create_workflow(
            name=template.get("name", template_name),
            org_id=org_id,
            metadata={"template": template_name, "parameters": parameters}
        )
        
        # Add tasks from template
        for task_def in template.get("tasks", []):
            task_name = task_def["name"].format(**parameters)
            function_name = task_def["function"]
            
            # Process arguments with parameters
            args = tuple(
                arg.format(**parameters) if isinstance(arg, str) else arg
                for arg in task_def.get("args", [])
            )
            
            kwargs = {}
            for key, value in task_def.get("kwargs", {}).items():
                if isinstance(value, str):
                    kwargs[key] = value.format(**parameters)
                else:
                    kwargs[key] = value
            
            self.orchestrator.add_task(
                workflow_id=workflow_id,
                task_name=task_name,
                function_name=function_name,
                args=args,
                kwargs=kwargs,
                dependencies=task_def.get("dependencies", []),
                priority=TaskPriority(task_def.get("priority", 3)),
                timeout_seconds=task_def.get("timeout_seconds"),
                max_retries=task_def.get("max_retries", 3)
            )
        
        logger.info(f"Created workflow {workflow_id} from template {template_name}")
        return workflow_id
    
    async def execute_template_workflow(self, template_name: str, org_id: str,
                                      parameters: Dict[str, Any] = None,
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create and execute workflow from template.
        
        Args:
            template_name: Template name
            org_id: Organization ID
            parameters: Template parameters
            context: Execution context
            
        Returns:
            Workflow execution results
        """
        workflow_id = await self.create_workflow_from_template(
            template_name, org_id, parameters
        )
        
        return await self.orchestrator.execute_workflow(workflow_id, context)