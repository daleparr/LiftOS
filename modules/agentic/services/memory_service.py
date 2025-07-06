"""
Memory Service Integration for Agentic Module

Provides integration with the LiftOS Memory Service for storing and
retrieving agent data, evaluation results, and test cases.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
import json

from ..models.agent_models import MarketingAgent
from ..models.evaluation_models import AgentEvaluationResult
from ..models.test_models import MarketingTestCase, TestResult

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Integration with LiftOS Memory Service.
    
    This service handles all data persistence operations for the
    Agentic module, including agents, evaluations, and test cases.
    """
    
    def __init__(self, memory_service_url: str):
        """Initialize memory service client."""
        self.base_url = memory_service_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Collection names for different data types
        self.collections = {
            "agents": "agentic_agents",
            "evaluations": "agentic_evaluations", 
            "test_cases": "agentic_test_cases",
            "test_results": "agentic_test_results",
            "scenarios": "agentic_scenarios"
        }
        
        logger.info(f"Memory Service client initialized for {self.base_url}")
    
    async def store_agent(self, agent: MarketingAgent) -> bool:
        """Store an agent in the memory service."""
        try:
            data = {
                "collection": self.collections["agents"],
                "key": agent.agent_id,
                "data": agent.dict(),
                "metadata": {
                    "type": "marketing_agent",
                    "agent_type": agent.agent_type.value,
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat()
                }
            }
            
            response = await self.client.post(f"{self.base_url}/store", json=data)
            response.raise_for_status()
            
            logger.debug(f"Stored agent {agent.agent_id} in memory service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store agent {agent.agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[MarketingAgent]:
        """Retrieve an agent from the memory service."""
        try:
            response = await self.client.get(
                f"{self.base_url}/retrieve/{self.collections['agents']}/{agent_id}"
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            return MarketingAgent(**data["data"])
            
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def list_agents(
        self,
        agent_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[MarketingAgent]:
        """List agents from the memory service."""
        try:
            params = {"collection": self.collections["agents"]}
            
            # Add filters
            filters = {}
            if agent_type:
                filters["agent_type"] = agent_type
            if active_only:
                filters["is_active"] = True
            
            if filters:
                params["filters"] = json.dumps(filters)
            
            response = await self.client.get(f"{self.base_url}/list", params=params)
            response.raise_for_status()
            
            data = response.json()
            agents = []
            
            for item in data.get("items", []):
                try:
                    agent = MarketingAgent(**item["data"])
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Failed to parse agent data: {e}")
                    continue
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent from the memory service."""
        try:
            response = await self.client.delete(
                f"{self.base_url}/delete/{self.collections['agents']}/{agent_id}"
            )
            response.raise_for_status()
            
            logger.debug(f"Deleted agent {agent_id} from memory service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            return False
    
    async def store_evaluation_result(self, result: AgentEvaluationResult) -> bool:
        """Store an evaluation result in the memory service."""
        try:
            data = {
                "collection": self.collections["evaluations"],
                "key": result.evaluation_id,
                "data": result.dict(),
                "metadata": {
                    "type": "evaluation_result",
                    "agent_id": result.agent_id,
                    "evaluation_date": result.evaluation_date.isoformat(),
                    "overall_score": result.overall_score,
                    "deployment_readiness": result.deployment_readiness.value
                }
            }
            
            response = await self.client.post(f"{self.base_url}/store", json=data)
            response.raise_for_status()
            
            logger.debug(f"Stored evaluation result {result.evaluation_id} in memory service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store evaluation result {result.evaluation_id}: {e}")
            return False
    
    async def get_evaluation_result(self, evaluation_id: str) -> Optional[AgentEvaluationResult]:
        """Retrieve an evaluation result from the memory service."""
        try:
            response = await self.client.get(
                f"{self.base_url}/retrieve/{self.collections['evaluations']}/{evaluation_id}"
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            return AgentEvaluationResult(**data["data"])
            
        except Exception as e:
            logger.error(f"Failed to get evaluation result {evaluation_id}: {e}")
            return None
    
    async def get_evaluations_since(
        self,
        since: datetime,
        agent_id: Optional[str] = None
    ) -> List[AgentEvaluationResult]:
        """Get evaluation results since a specific date."""
        try:
            params = {"collection": self.collections["evaluations"]}
            
            # Add filters
            filters = {
                "evaluation_date": {"$gte": since.isoformat()}
            }
            if agent_id:
                filters["agent_id"] = agent_id
            
            params["filters"] = json.dumps(filters)
            
            response = await self.client.get(f"{self.base_url}/list", params=params)
            response.raise_for_status()
            
            data = response.json()
            evaluations = []
            
            for item in data.get("items", []):
                try:
                    evaluation = AgentEvaluationResult(**item["data"])
                    evaluations.append(evaluation)
                except Exception as e:
                    logger.warning(f"Failed to parse evaluation data: {e}")
                    continue
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Failed to get evaluations since {since}: {e}")
            return []
    
    async def store_test_case(self, test_case: MarketingTestCase) -> bool:
        """Store a test case in the memory service."""
        try:
            data = {
                "collection": self.collections["test_cases"],
                "key": test_case.test_id,
                "data": test_case.dict(),
                "metadata": {
                    "type": "test_case",
                    "category": test_case.category,
                    "priority": test_case.priority.value,
                    "created_at": test_case.created_at.isoformat(),
                    "updated_at": test_case.updated_at.isoformat()
                }
            }
            
            response = await self.client.post(f"{self.base_url}/store", json=data)
            response.raise_for_status()
            
            logger.debug(f"Stored test case {test_case.test_id} in memory service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store test case {test_case.test_id}: {e}")
            return False
    
    async def get_test_case(self, test_case_id: str) -> Optional[MarketingTestCase]:
        """Retrieve a test case from the memory service."""
        try:
            response = await self.client.get(
                f"{self.base_url}/retrieve/{self.collections['test_cases']}/{test_case_id}"
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            return MarketingTestCase(**data["data"])
            
        except Exception as e:
            logger.error(f"Failed to get test case {test_case_id}: {e}")
            return None
    
    async def list_test_cases(
        self,
        category: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[MarketingTestCase]:
        """List test cases from the memory service."""
        try:
            params = {"collection": self.collections["test_cases"]}
            
            # Add filters
            filters = {}
            if category:
                filters["category"] = category
            if priority:
                filters["priority"] = priority
            
            if filters:
                params["filters"] = json.dumps(filters)
            
            response = await self.client.get(f"{self.base_url}/list", params=params)
            response.raise_for_status()
            
            data = response.json()
            test_cases = []
            
            for item in data.get("items", []):
                try:
                    test_case = MarketingTestCase(**item["data"])
                    test_cases.append(test_case)
                except Exception as e:
                    logger.warning(f"Failed to parse test case data: {e}")
                    continue
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to list test cases: {e}")
            return []
    
    async def store_test_result(self, result: TestResult) -> bool:
        """Store a test result in the memory service."""
        try:
            result_key = f"{result.test_id}_{result.start_time.isoformat()}"
            
            data = {
                "collection": self.collections["test_results"],
                "key": result_key,
                "data": result.dict(),
                "metadata": {
                    "type": "test_result",
                    "test_id": result.test_id,
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat(),
                    "overall_success": result.overall_success,
                    "score": result.score
                }
            }
            
            response = await self.client.post(f"{self.base_url}/store", json=data)
            response.raise_for_status()
            
            logger.debug(f"Stored test result {result_key} in memory service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store test result: {e}")
            return False
    
    async def get_test_result(self, test_id: str, agent_id: str) -> Optional[TestResult]:
        """Retrieve a test result from the memory service."""
        try:
            # Search for test results by test_id and agent_id
            params = {
                "collection": self.collections["test_results"],
                "filters": json.dumps({
                    "test_id": test_id,
                    "agent_outputs.agent_id": agent_id
                })
            }
            
            response = await self.client.get(f"{self.base_url}/list", params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                return None
            
            # Return the most recent result
            latest_item = max(items, key=lambda x: x["metadata"]["start_time"])
            return TestResult(**latest_item["data"])
            
        except Exception as e:
            logger.error(f"Failed to get test result for {test_id}/{agent_id}: {e}")
            return None
    
    async def search_data(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for data in a specific collection."""
        try:
            params = {
                "collection": self.collections.get(collection, collection),
                "filters": json.dumps(query),
                "limit": limit
            }
            
            response = await self.client.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("items", [])
            
        except Exception as e:
            logger.error(f"Failed to search {collection}: {e}")
            return []
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            collection_name = self.collections.get(collection, collection)
            response = await self.client.get(f"{self.base_url}/stats/{collection_name}")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection}: {e}")
            return {}
    
    async def cleanup_old_data(self, collection: str, older_than_days: int = 30) -> int:
        """Clean up old data from a collection."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            
            # This would depend on the memory service API supporting cleanup operations
            # For now, return 0 as placeholder
            logger.info(f"Cleanup requested for {collection} older than {older_than_days} days")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup {collection}: {e}")
            return 0
    
    async def close(self) -> None:
        """Close the HTTP client."""
        try:
            await self.client.aclose()
            logger.debug("Memory service client closed")
        except Exception as e:
            logger.error(f"Error closing memory service client: {e}")