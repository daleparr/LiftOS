"""
Intelligence SDK Client
Main client for interacting with LiftOS intelligence services
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import httpx
from datetime import datetime

from ..models.learning import (
    LearningType, LearningRequest, LearningResponse, Pattern, KnowledgeItem
)
from ..models.decision import (
    DecisionType, DecisionRequest, DecisionResponse, Decision, 
    Recommendation, AutomatedAction, DecisionOutcome
)


class IntelligenceClient:
    """Main client for LiftOS intelligence services"""
    
    def __init__(
        self,
        intelligence_url: str = "http://localhost:8009",
        feedback_url: str = "http://localhost:8010",
        timeout: float = 60.0,
        api_key: Optional[str] = None
    ):
        self.intelligence_url = intelligence_url.rstrip('/')
        self.feedback_url = feedback_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key
        
        # HTTP client configuration
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    # Learning Methods
    async def start_learning(
        self,
        learning_type: LearningType,
        name: str,
        data_sources: List[str],
        user_id: str,
        organization_id: str,
        description: Optional[str] = None,
        target_variables: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> LearningResponse:
        """Start a learning process"""
        request = LearningRequest(
            learning_type=learning_type,
            name=name,
            description=description,
            data_sources=data_sources,
            target_variables=target_variables or [],
            parameters=parameters or {},
            user_id=user_id,
            organization_id=organization_id
        )
        
        response = await self.client.post(
            f"{self.intelligence_url}/api/v1/learning/start",
            json=request.dict()
        )
        response.raise_for_status()
        
        data = response.json()
        return LearningResponse(**data)
    
    async def get_learning_status(self, learning_id: str) -> Dict[str, Any]:
        """Get status of learning process"""
        response = await self.client.get(
            f"{self.intelligence_url}/api/v1/learning/{learning_id}/status"
        )
        response.raise_for_status()
        
        return response.json()["data"]
    
    async def wait_for_learning_completion(
        self, 
        learning_id: str, 
        poll_interval: float = 5.0,
        max_wait_time: float = 1800.0  # 30 minutes
    ) -> Dict[str, Any]:
        """Wait for learning process to complete"""
        start_time = datetime.utcnow()
        
        while True:
            status_data = await self.get_learning_status(learning_id)
            
            if status_data["status"] in ["completed", "failed"]:
                return status_data
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_wait_time:
                raise TimeoutError(f"Learning process {learning_id} did not complete within {max_wait_time} seconds")
            
            await asyncio.sleep(poll_interval)
    
    # Decision Methods
    async def request_decision(
        self,
        decision_type: DecisionType,
        title: str,
        user_id: str,
        organization_id: str,
        domain: str,
        description: Optional[str] = None,
        context: Dict[str, Any] = None,
        constraints: List[Dict[str, Any]] = None,
        objectives: List[str] = None,
        time_horizon: Optional[str] = None,
        risk_tolerance: str = "medium",
        confidence_threshold: float = 0.7
    ) -> DecisionResponse:
        """Request a decision from the intelligence system"""
        request = DecisionRequest(
            decision_type=decision_type,
            title=title,
            description=description,
            context=context or {},
            constraints=constraints or [],
            objectives=objectives or [],
            time_horizon=time_horizon,
            risk_tolerance=risk_tolerance,
            confidence_threshold=confidence_threshold,
            user_id=user_id,
            organization_id=organization_id,
            domain=domain
        )
        
        response = await self.client.post(
            f"{self.intelligence_url}/api/v1/decisions/request",
            json=request.dict()
        )
        response.raise_for_status()
        
        data = response.json()
        return DecisionResponse(**data)
    
    async def get_decision_result(self, decision_id: str) -> Dict[str, Any]:
        """Get result of decision process"""
        response = await self.client.get(
            f"{self.intelligence_url}/api/v1/decisions/{decision_id}/result"
        )
        response.raise_for_status()
        
        return response.json()["data"]
    
    async def wait_for_decision_completion(
        self,
        decision_id: str,
        poll_interval: float = 2.0,
        max_wait_time: float = 300.0  # 5 minutes
    ) -> Dict[str, Any]:
        """Wait for decision process to complete"""
        start_time = datetime.utcnow()
        
        while True:
            result_data = await self.get_decision_result(decision_id)
            
            if result_data["status"] in ["completed", "failed"]:
                return result_data
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_wait_time:
                raise TimeoutError(f"Decision process {decision_id} did not complete within {max_wait_time} seconds")
            
            await asyncio.sleep(poll_interval)
    
    # Pattern and Knowledge Methods
    async def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Pattern]:
        """Get discovered patterns"""
        params = {"limit": limit}
        if pattern_type:
            params["pattern_type"] = pattern_type
        
        response = await self.client.get(
            f"{self.intelligence_url}/api/v1/patterns",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()["data"]
        return [Pattern(**pattern) for pattern in data["patterns"]]
    
    async def get_knowledge_base(
        self,
        domain: Optional[str] = None,
        limit: int = 50
    ) -> List[KnowledgeItem]:
        """Get knowledge base items"""
        params = {"limit": limit}
        if domain:
            params["domain"] = domain
        
        response = await self.client.get(
            f"{self.intelligence_url}/api/v1/knowledge",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()["data"]
        return [KnowledgeItem(**item) for item in data["knowledge_items"]]
    
    # Feedback Methods
    async def report_outcome(
        self,
        decision_id: str,
        actual_impact: Dict[str, float],
        success_metrics: Dict[str, float] = None,
        unexpected_consequences: List[str] = None,
        user_satisfaction: Optional[float] = None,
        business_value: Optional[float] = None,
        measurement_period: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Report decision outcome for learning"""
        request_data = {
            "decision_id": decision_id,
            "actual_impact": actual_impact,
            "success_metrics": success_metrics or {},
            "unexpected_consequences": unexpected_consequences or [],
            "user_satisfaction": user_satisfaction,
            "business_value": business_value,
            "measurement_period": measurement_period,
            "notes": notes
        }
        
        response = await self.client.post(
            f"{self.feedback_url}/api/v1/outcomes/report",
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()["data"]
    
    async def provide_learning_feedback(
        self,
        learning_id: str,
        model_id: str,
        accuracy_feedback: Dict[str, float] = None,
        pattern_validation: Dict[str, bool] = None,
        insight_usefulness: Dict[str, float] = None,
        recommendation_adoption: Dict[str, bool] = None,
        improvement_suggestions: List[str] = None
    ) -> Dict[str, Any]:
        """Provide feedback on learning results"""
        request_data = {
            "learning_id": learning_id,
            "model_id": model_id,
            "accuracy_feedback": accuracy_feedback or {},
            "pattern_validation": pattern_validation or {},
            "insight_usefulness": insight_usefulness or {},
            "recommendation_adoption": recommendation_adoption or {},
            "improvement_suggestions": improvement_suggestions or []
        }
        
        response = await self.client.post(
            f"{self.feedback_url}/api/v1/learning/feedback",
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()["data"]
    
    async def analyze_feedback(
        self,
        time_period: str = "30d",
        model_ids: Optional[List[str]] = None,
        decision_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze feedback patterns and trends"""
        params = {"time_period": time_period}
        if model_ids:
            params["model_ids"] = model_ids
        if decision_types:
            params["decision_types"] = decision_types
        if domains:
            params["domains"] = domains
        
        response = await self.client.get(
            f"{self.feedback_url}/api/v1/analysis/feedback",
            params=params
        )
        response.raise_for_status()
        
        return response.json()["data"]
    
    # Convenience Methods
    async def learn_and_decide(
        self,
        learning_type: LearningType,
        decision_type: DecisionType,
        name: str,
        data_sources: List[str],
        user_id: str,
        organization_id: str,
        domain: str,
        context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Combined learning and decision process"""
        # Start learning
        learning_response = await self.start_learning(
            learning_type=learning_type,
            name=f"Learning for {name}",
            data_sources=data_sources,
            user_id=user_id,
            organization_id=organization_id
        )
        
        # Wait for learning completion
        learning_result = await self.wait_for_learning_completion(
            learning_response.learning_id
        )
        
        # Request decision based on learning
        decision_response = await self.request_decision(
            decision_type=decision_type,
            title=f"Decision for {name}",
            user_id=user_id,
            organization_id=organization_id,
            domain=domain,
            context=context or {}
        )
        
        # Wait for decision completion
        decision_result = await self.wait_for_decision_completion(
            decision_response.decision_id
        )
        
        return learning_result, decision_result
    
    async def get_intelligence_summary(
        self,
        organization_id: str,
        time_period: str = "7d"
    ) -> Dict[str, Any]:
        """Get comprehensive intelligence summary"""
        # Get patterns
        patterns = await self.get_patterns(limit=20)
        
        # Get knowledge base
        knowledge = await self.get_knowledge_base(limit=20)
        
        # Get feedback analysis
        feedback_analysis = await self.analyze_feedback(time_period=time_period)
        
        return {
            "patterns": {
                "total": len(patterns),
                "by_type": {},
                "recent": [p.dict() for p in patterns[:5]]
            },
            "knowledge": {
                "total": len(knowledge),
                "by_domain": {},
                "recent": [k.dict() for k in knowledge[:5]]
            },
            "feedback_analysis": feedback_analysis,
            "summary": {
                "active_learning_processes": 0,  # Would need to track this
                "pending_decisions": 0,  # Would need to track this
                "avg_confidence": feedback_analysis.get("learning_metrics", {}).get("avg_accuracy_feedback", 0),
                "improvement_trend": "stable"  # Would calculate from trends
            }
        }


# Convenience function for creating client
def create_intelligence_client(
    intelligence_url: str = "http://localhost:8009",
    feedback_url: str = "http://localhost:8010",
    api_key: Optional[str] = None
) -> IntelligenceClient:
    """Create an intelligence client with default configuration"""
    return IntelligenceClient(
        intelligence_url=intelligence_url,
        feedback_url=feedback_url,
        api_key=api_key
    )