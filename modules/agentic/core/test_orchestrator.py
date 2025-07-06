"""
Test Orchestrator for Agentic Module

Orchestrates test execution for marketing agents, managing test cases,
scenarios, and coordinating with the evaluation engine.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from ..models.agent_models import MarketingAgent, AgentCapability
from ..models.test_models import (
    MarketingTestCase, TestResult, TestScenario, TestStep,
    TestStatus, TestPriority, MarketingScenarioType, SuccessCriteria
)
from ..models.evaluation_models import AgentEvaluationResult
from ..core.agent_manager import AgentManager
from ..core.evaluation_engine import EvaluationEngine
from ..services.memory_service import MemoryService
from ..utils.config import AgenticConfig
from ..test_cases.marketing_test_library import MarketingTestLibrary

logger = logging.getLogger(__name__)


class TestOrchestrator:
    """
    Orchestrates test execution for marketing agents.
    
    This class manages test cases, coordinates test execution,
    and integrates with the evaluation engine for comprehensive
    agent assessment.
    """
    
    def __init__(
        self,
        agent_manager: AgentManager,
        evaluation_engine: EvaluationEngine,
        memory_service: MemoryService,
        config: AgenticConfig
    ):
        """Initialize the test orchestrator."""
        self.agent_manager = agent_manager
        self.evaluation_engine = evaluation_engine
        self.memory_service = memory_service
        self.config = config
        self.test_library = MarketingTestLibrary()
        
        # Test execution tracking
        self._running_tests: Dict[str, asyncio.Task] = {}
        self._test_results: Dict[str, TestResult] = {}
        
        # Test case cache
        self._test_case_cache: Dict[str, MarketingTestCase] = {}
        self._scenario_cache: Dict[str, TestScenario] = {}
        
        logger.info("Test Orchestrator initialized")
    
    async def load_default_test_cases(self) -> None:
        """Load default test cases from the library."""
        try:
            default_test_cases = await self.test_library.get_default_test_cases()
            
            for test_case in default_test_cases:
                # Check if test case already exists
                existing = await self.get_test_case(test_case.test_id)
                if not existing:
                    await self.register_test_case(test_case)
                    logger.info(f"Loaded default test case: {test_case.name}")
                else:
                    logger.debug(f"Test case {test_case.name} already exists, skipping")
            
            logger.info(f"Loaded {len(default_test_cases)} default test cases")
            
        except Exception as e:
            logger.error(f"Failed to load default test cases: {e}")
            raise
    
    async def register_test_case(self, test_case: MarketingTestCase) -> MarketingTestCase:
        """Register a new test case."""
        try:
            # Validate test case
            await self._validate_test_case(test_case)
            
            # Generate ID if not provided
            if not test_case.test_id:
                test_case.test_id = f"test_{uuid.uuid4().hex[:8]}"
            
            # Set timestamps
            test_case.created_at = datetime.utcnow()
            test_case.updated_at = datetime.utcnow()
            
            # Store in memory service
            await self.memory_service.store_test_case(test_case)
            
            # Add to cache
            self._test_case_cache[test_case.test_id] = test_case
            
            logger.info(f"Registered test case: {test_case.name} ({test_case.test_id})")
            return test_case
            
        except Exception as e:
            logger.error(f"Failed to register test case {test_case.name}: {e}")
            raise
    
    async def get_test_case(self, test_case_id: str) -> Optional[MarketingTestCase]:
        """Get a test case by ID."""
        try:
            # Check cache first
            if test_case_id in self._test_case_cache:
                return self._test_case_cache[test_case_id]
            
            # Load from memory service
            test_case = await self.memory_service.get_test_case(test_case_id)
            if test_case:
                self._test_case_cache[test_case_id] = test_case
            
            return test_case
            
        except Exception as e:
            logger.error(f"Failed to get test case {test_case_id}: {e}")
            return None
    
    async def list_test_cases(
        self,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        agent_type: Optional[str] = None
    ) -> List[MarketingTestCase]:
        """List test cases with optional filtering."""
        try:
            test_cases = await self.memory_service.list_test_cases()
            
            # Apply filters
            filtered_cases = []
            for test_case in test_cases:
                # Filter by category
                if category and test_case.category != category:
                    continue
                
                # Filter by priority
                if priority and test_case.priority.value != priority:
                    continue
                
                # Filter by agent type (check required capabilities)
                if agent_type:
                    # This would need agent type to capability mapping
                    pass
                
                filtered_cases.append(test_case)
            
            # Update cache
            for test_case in filtered_cases:
                self._test_case_cache[test_case.test_id] = test_case
            
            return filtered_cases
            
        except Exception as e:
            logger.error(f"Failed to list test cases: {e}")
            return []
    
    async def list_scenarios(
        self,
        scenario_type: Optional[str] = None
    ) -> List[TestScenario]:
        """List available test scenarios."""
        try:
            scenarios = await self.test_library.get_scenarios(scenario_type)
            
            # Update cache
            for scenario in scenarios:
                self._scenario_cache[scenario.scenario_id] = scenario
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Failed to list scenarios: {e}")
            return []
    
    async def run_test_case(
        self,
        test_case_id: str,
        agent_id: str,
        background: bool = False
    ) -> TestResult:
        """Run a test case against an agent."""
        try:
            # Get test case and agent
            test_case = await self.get_test_case(test_case_id)
            if not test_case:
                raise ValueError(f"Test case {test_case_id} not found")
            
            agent = await self.agent_manager.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Validate agent compatibility
            if not test_case.validate_agent_compatibility(agent.capabilities):
                raise ValueError(f"Agent {agent_id} lacks required capabilities for test {test_case_id}")
            
            # Create test result
            test_result = TestResult(
                test_id=test_case_id,
                status=TestStatus.PENDING,
                start_time=datetime.utcnow()
            )
            
            # Execute test
            if background:
                # Run in background
                task = asyncio.create_task(
                    self._execute_test_case(test_case, agent, test_result)
                )
                self._running_tests[f"{test_case_id}_{agent_id}"] = task
                test_result.status = TestStatus.RUNNING
                return test_result
            else:
                # Run synchronously
                return await self._execute_test_case(test_case, agent, test_result)
            
        except Exception as e:
            logger.error(f"Failed to run test case {test_case_id} for agent {agent_id}: {e}")
            raise
    
    async def run_test_suite(
        self,
        agent_id: str,
        test_case_ids: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[TestResult]:
        """Run a suite of test cases against an agent."""
        try:
            # Get test cases
            if test_case_ids:
                test_cases = []
                for test_id in test_case_ids:
                    test_case = await self.get_test_case(test_id)
                    if test_case:
                        test_cases.append(test_case)
            else:
                test_cases = await self.list_test_cases(category=category)
            
            if not test_cases:
                return []
            
            # Get agent
            agent = await self.agent_manager.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Run tests in parallel (with concurrency limit)
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tests)
            
            async def run_single_test(test_case: MarketingTestCase) -> TestResult:
                async with semaphore:
                    return await self.run_test_case(test_case.test_id, agent_id)
            
            # Execute all tests
            tasks = [run_single_test(tc) for tc in test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = [r for r in results if isinstance(r, TestResult)]
            
            logger.info(f"Completed test suite for agent {agent_id}: {len(valid_results)} tests")
            return valid_results
            
        except Exception as e:
            logger.error(f"Failed to run test suite for agent {agent_id}: {e}")
            raise
    
    async def get_test_result(self, test_id: str, agent_id: str) -> Optional[TestResult]:
        """Get test result for a specific test and agent."""
        try:
            result_key = f"{test_id}_{agent_id}"
            
            # Check local cache
            if result_key in self._test_results:
                return self._test_results[result_key]
            
            # Load from memory service
            result = await self.memory_service.get_test_result(test_id, agent_id)
            if result:
                self._test_results[result_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get test result for {test_id}/{agent_id}: {e}")
            return None
    
    async def get_running_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently running tests."""
        running_info = {}
        
        for test_key, task in self._running_tests.items():
            if not task.done():
                running_info[test_key] = {
                    "status": "running",
                    "started": task.get_name() if hasattr(task, 'get_name') else "unknown"
                }
            else:
                # Clean up completed tasks
                del self._running_tests[test_key]
        
        return running_info
    
    async def cancel_test(self, test_id: str, agent_id: str) -> bool:
        """Cancel a running test."""
        try:
            test_key = f"{test_id}_{agent_id}"
            
            if test_key in self._running_tests:
                task = self._running_tests[test_key]
                if not task.done():
                    task.cancel()
                    
                    # Update test result
                    if test_key in self._test_results:
                        result = self._test_results[test_key]
                        result.status = TestStatus.CANCELLED
                        result.end_time = datetime.utcnow()
                        await self.memory_service.store_test_result(result)
                    
                    logger.info(f"Cancelled test {test_id} for agent {agent_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel test {test_id}/{agent_id}: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Cancel all running tests
            for task in self._running_tests.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._running_tests:
                await asyncio.gather(
                    *self._running_tests.values(),
                    return_exceptions=True
                )
            
            # Clear caches
            self._running_tests.clear()
            self._test_results.clear()
            self._test_case_cache.clear()
            self._scenario_cache.clear()
            
            logger.info("Test Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Test Orchestrator cleanup: {e}")
    
    async def _execute_test_case(
        self,
        test_case: MarketingTestCase,
        agent: MarketingAgent,
        test_result: TestResult
    ) -> TestResult:
        """Execute a single test case."""
        try:
            logger.info(f"Executing test case {test_case.name} for agent {agent.name}")
            
            test_result.status = TestStatus.RUNNING
            test_result.start_time = datetime.utcnow()
            
            # Execute test steps
            step_results = []
            failed_steps = []
            
            for step in test_case.steps:
                try:
                    step_result = await self._execute_test_step(step, agent, test_case)
                    step_results.append(step_result)
                    
                    if not step_result.get("success", False):
                        failed_steps.append(step.step_id)
                        
                        # Stop on critical step failure
                        if step.retry_count == 0:
                            break
                            
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {e}")
                    step_results.append({
                        "step_id": step.step_id,
                        "success": False,
                        "error": str(e)
                    })
                    failed_steps.append(step.step_id)
                    break
            
            # Evaluate success criteria
            success_criteria_met = await self._evaluate_success_criteria(
                test_case.success_criteria, step_results, agent
            )
            
            # Calculate overall success
            overall_success = (
                len(failed_steps) == 0 and
                all(success_criteria_met)
            )
            
            # Calculate score
            score = self._calculate_test_score(success_criteria_met, step_results)
            
            # Update test result
            test_result.status = TestStatus.COMPLETED if overall_success else TestStatus.FAILED
            test_result.end_time = datetime.utcnow()
            test_result.duration_seconds = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.success_criteria_met = success_criteria_met
            test_result.overall_success = overall_success
            test_result.score = score
            test_result.step_results = step_results
            test_result.failed_steps = failed_steps
            
            # Store result
            await self.memory_service.store_test_result(test_result)
            
            # Update test case statistics
            test_case.update_execution_stats(test_result.duration_seconds, overall_success)
            await self.memory_service.store_test_case(test_case)
            
            # Cache result
            result_key = f"{test_case.test_id}_{agent.agent_id}"
            self._test_results[result_key] = test_result
            
            logger.info(f"Test case {test_case.name} completed with score {score:.3f}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to execute test case {test_case.test_id}: {e}")
            
            # Update test result with error
            test_result.status = TestStatus.FAILED
            test_result.end_time = datetime.utcnow()
            test_result.errors.append(str(e))
            
            return test_result
    
    async def _execute_test_step(
        self,
        step: TestStep,
        agent: MarketingAgent,
        test_case: MarketingTestCase
    ) -> Dict[str, Any]:
        """Execute a single test step."""
        try:
            logger.debug(f"Executing step {step.step_id}: {step.name}")
            
            # Prepare step data
            step_data = {
                "action": step.action,
                "parameters": step.parameters,
                "test_case_id": test_case.test_id,
                "step_id": step.step_id
            }
            
            # Execute step using agent manager
            result = await self.agent_manager.execute_agent_task(
                agent.agent_id,
                step_data,
                step.timeout_seconds
            )
            
            # Validate expected output if provided
            success = True
            validation_errors = []
            
            if step.expected_output:
                success, validation_errors = self._validate_step_output(
                    result, step.expected_output
                )
            
            return {
                "step_id": step.step_id,
                "success": success,
                "result": result,
                "validation_errors": validation_errors,
                "execution_time": result.get("execution_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to execute step {step.step_id}: {e}")
            return {
                "step_id": step.step_id,
                "success": False,
                "error": str(e)
            }
    
    async def _evaluate_success_criteria(
        self,
        criteria: List[SuccessCriteria],
        step_results: List[Dict[str, Any]],
        agent: MarketingAgent
    ) -> List[bool]:
        """Evaluate success criteria against test results."""
        criteria_results = []
        
        for criterion in criteria:
            try:
                # Extract metric value from results
                metric_value = self._extract_metric_value(
                    criterion.metric_name, step_results, agent
                )
                
                if metric_value is None:
                    criteria_results.append(False)
                    continue
                
                # Evaluate criterion
                success = self._evaluate_criterion(
                    metric_value, criterion.operator, criterion.threshold
                )
                criteria_results.append(success)
                
            except Exception as e:
                logger.error(f"Failed to evaluate criterion {criterion.metric_name}: {e}")
                criteria_results.append(False)
        
        return criteria_results
    
    def _extract_metric_value(
        self,
        metric_name: str,
        step_results: List[Dict[str, Any]],
        agent: MarketingAgent
    ) -> Optional[float]:
        """Extract metric value from test results."""
        # This would be implemented based on specific metrics
        # For now, return placeholder values
        
        if metric_name == "accuracy":
            return 0.85
        elif metric_name == "response_time":
            return 15.0
        elif metric_name == "success_rate":
            return len([r for r in step_results if r.get("success", False)]) / len(step_results)
        
        return None
    
    def _evaluate_criterion(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a single success criterion."""
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001
        elif operator == "!=":
            return abs(value - threshold) >= 0.001
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        else:
            return False
    
    def _calculate_test_score(
        self,
        success_criteria_met: List[bool],
        step_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall test score."""
        if not success_criteria_met and not step_results:
            return 0.0
        
        # Weight criteria and step success equally
        criteria_score = sum(success_criteria_met) / len(success_criteria_met) if success_criteria_met else 0.0
        step_score = sum(1 for r in step_results if r.get("success", False)) / len(step_results) if step_results else 0.0
        
        return (criteria_score + step_score) / 2.0
    
    def _validate_step_output(
        self,
        actual_output: Dict[str, Any],
        expected_output: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate step output against expected output."""
        errors = []
        
        for key, expected_value in expected_output.items():
            if key not in actual_output:
                errors.append(f"Missing expected key: {key}")
                continue
            
            actual_value = actual_output[key]
            if actual_value != expected_value:
                errors.append(f"Key {key}: expected {expected_value}, got {actual_value}")
        
        return len(errors) == 0, errors
    
    async def _validate_test_case(self, test_case: MarketingTestCase) -> None:
        """Validate test case configuration."""
        if not test_case.name or not test_case.description:
            raise ValueError("Test case name and description are required")
        
        if not test_case.steps:
            raise ValueError("Test case must have at least one step")
        
        if not test_case.success_criteria:
            raise ValueError("Test case must have success criteria")
        
        if not test_case.required_capabilities:
            raise ValueError("Test case must specify required capabilities")
        
        # Validate step dependencies
        step_ids = {step.step_id for step in test_case.steps}
        for step in test_case.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on non-existent step {dep}")