"""
LiftOS User Analytics Service
Comprehensive user behavior tracking and analysis service.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import json

# Import models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain


from shared.models.business import BusinessMetric, MetricType, MetricFrequency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserEvent:
    """User interaction event"""
    def __init__(self, user_id: str, event_type: str, event_data: Dict[str, Any]):
        self.id = f"{user_id}_{datetime.utcnow().timestamp()}"
        self.user_id = user_id
        self.event_type = event_type
        self.event_data = event_data
        self.timestamp = datetime.utcnow()

class UserSession:
    """User session tracking"""
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.events: List[UserEvent] = []
        self.decisions_made = 0
        self.features_used: set = set()
        self.workflows_completed = 0
        self.satisfaction_score: Optional[float] = None

class DecisionJourney:
    """Track user's decision-making journey"""
    def __init__(self, user_id: str, decision_id: str):
        self.user_id = user_id
        self.decision_id = decision_id
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.steps: List[Dict[str, Any]] = []
        self.data_sources_accessed: List[str] = []
        self.insights_viewed: List[str] = []
        self.recommendations_received: List[str] = []
        self.final_decision: Optional[Dict[str, Any]] = None
        self.confidence_score: Optional[float] = None
        self.outcome: Optional[str] = None

class UserBehaviorAnalytics:
    """Core user behavior analytics engine"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.completed_sessions: List[UserSession] = []
        self.decision_journeys: Dict[str, DecisionJourney] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.feature_usage: Dict[str, int] = defaultdict(int)
        self.workflow_analytics: Dict[str, List[float]] = defaultdict(list)
        
        # Real-time analytics
        self.recent_events = deque(maxlen=1000)
        self.user_activity_buffer: Dict[str, List[UserEvent]] = defaultdict(list)
        
    async def track_user_event(self, user_id: str, event_type: str, event_data: Dict[str, Any]) -> UserEvent:
        """Track a user interaction event"""
        try:
            event = UserEvent(user_id, event_type, event_data)
            
            # Add to recent events
            self.recent_events.append(event)
            self.user_activity_buffer[user_id].append(event)
            
            # Update active session if exists
            session_id = event_data.get('session_id')
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.events.append(event)
                
                # Update session metrics based on event type
                await self._update_session_metrics(session, event)
            
            # Track feature usage
            feature = event_data.get('feature')
            if feature:
                self.feature_usage[feature] += 1
            
            # Process specific event types
            await self._process_event_type(event)
            
            logger.info(f"Tracked event: {event_type} for user {user_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error tracking user event: {e}")
            raise
    
    async def start_user_session(self, user_id: str, session_data: Dict[str, Any]) -> UserSession:
        """Start a new user session"""
        try:
            session_id = session_data.get('session_id', f"{user_id}_{datetime.utcnow().timestamp()}")
            
            session = UserSession(user_id, session_id)
            self.active_sessions[session_id] = session
            
            # Initialize user profile if new user
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    'first_seen': datetime.utcnow(),
                    'total_sessions': 0,
                    'total_decisions': 0,
                    'preferred_features': [],
                    'skill_level': 'beginner',
                    'satisfaction_history': []
                }
            
            self.user_profiles[user_id]['total_sessions'] += 1
            
            logger.info(f"Started session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error starting user session: {e}")
            raise
    
    async def end_user_session(self, session_id: str, session_data: Dict[str, Any]) -> UserSession:
        """End a user session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.utcnow()
            session.satisfaction_score = session_data.get('satisfaction_score')
            
            # Calculate session metrics
            session_duration = (session.end_time - session.start_time).total_seconds()
            
            # Update user profile
            user_profile = self.user_profiles[session.user_id]
            user_profile['total_decisions'] += session.decisions_made
            
            if session.satisfaction_score:
                user_profile['satisfaction_history'].append(session.satisfaction_score)
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session {session_id}, duration: {session_duration:.1f}s")
            return session
            
        except Exception as e:
            logger.error(f"Error ending user session: {e}")
            raise
    
    async def start_decision_journey(self, user_id: str, decision_data: Dict[str, Any]) -> DecisionJourney:
        """Start tracking a decision journey"""
        try:
            decision_id = decision_data.get('decision_id', f"decision_{datetime.utcnow().timestamp()}")
            
            journey = DecisionJourney(user_id, decision_id)
            self.decision_journeys[decision_id] = journey
            
            # Add initial step
            initial_step = {
                'step_type': 'start',
                'timestamp': journey.start_time,
                'context': decision_data.get('context', {})
            }
            journey.steps.append(initial_step)
            
            logger.info(f"Started decision journey {decision_id} for user {user_id}")
            return journey
            
        except Exception as e:
            logger.error(f"Error starting decision journey: {e}")
            raise
    
    async def track_decision_step(self, decision_id: str, step_data: Dict[str, Any]):
        """Track a step in the decision journey"""
        try:
            if decision_id not in self.decision_journeys:
                raise ValueError(f"Decision journey {decision_id} not found")
            
            journey = self.decision_journeys[decision_id]
            
            step = {
                'step_type': step_data.get('step_type'),
                'timestamp': datetime.utcnow(),
                'data': step_data.get('data', {}),
                'duration': step_data.get('duration', 0)
            }
            journey.steps.append(step)
            
            # Track specific step types
            step_type = step_data.get('step_type')
            if step_type == 'data_access':
                data_source = step_data.get('data', {}).get('source')
                if data_source:
                    journey.data_sources_accessed.append(data_source)
            elif step_type == 'insight_view':
                insight_id = step_data.get('data', {}).get('insight_id')
                if insight_id:
                    journey.insights_viewed.append(insight_id)
            elif step_type == 'recommendation_received':
                recommendation_id = step_data.get('data', {}).get('recommendation_id')
                if recommendation_id:
                    journey.recommendations_received.append(recommendation_id)
            
            logger.info(f"Tracked decision step: {step_type} for journey {decision_id}")
            
        except Exception as e:
            logger.error(f"Error tracking decision step: {e}")
            raise
    
    async def complete_decision_journey(self, decision_id: str, completion_data: Dict[str, Any]) -> DecisionJourney:
        """Complete a decision journey"""
        try:
            if decision_id not in self.decision_journeys:
                raise ValueError(f"Decision journey {decision_id} not found")
            
            journey = self.decision_journeys[decision_id]
            journey.end_time = datetime.utcnow()
            journey.final_decision = completion_data.get('decision')
            journey.confidence_score = completion_data.get('confidence_score')
            
            # Add completion step
            completion_step = {
                'step_type': 'complete',
                'timestamp': journey.end_time,
                'decision': journey.final_decision,
                'confidence': journey.confidence_score
            }
            journey.steps.append(completion_step)
            
            # Calculate journey metrics
            journey_duration = (journey.end_time - journey.start_time).total_seconds()
            
            # Update user profile
            user_profile = self.user_profiles[journey.user_id]
            user_profile['total_decisions'] += 1
            
            logger.info(f"Completed decision journey {decision_id}, duration: {journey_duration:.1f}s")
            return journey
            
        except Exception as e:
            logger.error(f"Error completing decision journey: {e}")
            raise
    
    async def _update_session_metrics(self, session: UserSession, event: UserEvent):
        """Update session metrics based on event"""
        try:
            event_type = event.event_type
            
            if event_type == 'decision_made':
                session.decisions_made += 1
            elif event_type == 'feature_used':
                feature = event.event_data.get('feature')
                if feature:
                    session.features_used.add(feature)
            elif event_type == 'workflow_completed':
                session.workflows_completed += 1
                
                # Track workflow completion time
                workflow_name = event.event_data.get('workflow_name')
                completion_time = event.event_data.get('completion_time', 0)
                if workflow_name and completion_time:
                    self.workflow_analytics[workflow_name].append(completion_time)
            
        except Exception as e:
            logger.error(f"Error updating session metrics: {e}")
    
    async def _process_event_type(self, event: UserEvent):
        """Process specific event types for analytics"""
        try:
            event_type = event.event_type
            
            if event_type == 'page_view':
                await self._track_page_analytics(event)
            elif event_type == 'feature_interaction':
                await self._track_feature_analytics(event)
            elif event_type == 'error_encountered':
                await self._track_error_analytics(event)
            elif event_type == 'feedback_submitted':
                await self._track_feedback_analytics(event)
            
        except Exception as e:
            logger.error(f"Error processing event type: {e}")
    
    async def _track_page_analytics(self, event: UserEvent):
        """Track page view analytics"""
        try:
            page = event.event_data.get('page')
            duration = event.event_data.get('duration', 0)
            
            # Track page popularity and engagement
            if page:
                self.feature_usage[f"page_{page}"] += 1
            
        except Exception as e:
            logger.error(f"Error tracking page analytics: {e}")
    
    async def _track_feature_analytics(self, event: UserEvent):
        """Track feature interaction analytics"""
        try:
            feature = event.event_data.get('feature')
            interaction_type = event.event_data.get('interaction_type')
            
            if feature and interaction_type:
                self.feature_usage[f"{feature}_{interaction_type}"] += 1
            
        except Exception as e:
            logger.error(f"Error tracking feature analytics: {e}")
    
    async def _track_error_analytics(self, event: UserEvent):
        """Track error analytics"""
        try:
            error_type = event.event_data.get('error_type')
            error_context = event.event_data.get('context', {})
            
            # Track error patterns
            if error_type:
                self.feature_usage[f"error_{error_type}"] += 1
            
        except Exception as e:
            logger.error(f"Error tracking error analytics: {e}")
    
    async def _track_feedback_analytics(self, event: UserEvent):
        """Track user feedback analytics"""
        try:
            feedback_type = event.event_data.get('feedback_type')
            rating = event.event_data.get('rating')
            
            # Update user satisfaction in profile
            user_profile = self.user_profiles[event.user_id]
            if rating:
                user_profile['satisfaction_history'].append(rating)
            
        except Exception as e:
            logger.error(f"Error tracking feedback analytics: {e}")
    
    async def get_user_behavior_summary(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user behavior analytics summary"""
        try:
            if user_id:
                # User-specific summary
                user_profile = self.user_profiles.get(user_id, {})
                user_events = [e for e in self.recent_events if e.user_id == user_id]
                
                summary = {
                    "user_id": user_id,
                    "profile": user_profile,
                    "recent_activity": len(user_events),
                    "active_sessions": len([s for s in self.active_sessions.values() if s.user_id == user_id])
                }
            else:
                # Overall summary
                total_users = len(self.user_profiles)
                active_sessions = len(self.active_sessions)
                total_events = len(self.recent_events)
                
                # Calculate average satisfaction
                all_satisfaction = []
                for profile in self.user_profiles.values():
                    all_satisfaction.extend(profile.get('satisfaction_history', []))
                
                avg_satisfaction = sum(all_satisfaction) / len(all_satisfaction) if all_satisfaction else 0
                
                # Top features
                top_features = sorted(self.feature_usage.items(), key=lambda x: x[1], reverse=True)[:10]
                
                summary = {
                    "total_users": total_users,
                    "active_sessions": active_sessions,
                    "total_events": total_events,
                    "average_satisfaction": avg_satisfaction,
                    "top_features": top_features,
                    "workflow_performance": dict(self.workflow_analytics)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating user behavior summary: {e}")
            raise
    
    async def get_decision_analytics(self) -> Dict[str, Any]:
        """Get decision journey analytics"""
        try:
            completed_journeys = [j for j in self.decision_journeys.values() if j.end_time]
            
            if not completed_journeys:
                return {"message": "No completed decision journeys"}
            
            # Calculate metrics
            total_journeys = len(completed_journeys)
            avg_duration = sum((j.end_time - j.start_time).total_seconds() for j in completed_journeys) / total_journeys
            avg_steps = sum(len(j.steps) for j in completed_journeys) / total_journeys
            avg_confidence = sum(j.confidence_score or 0 for j in completed_journeys) / total_journeys
            
            # Data source usage
            data_source_usage = defaultdict(int)
            for journey in completed_journeys:
                for source in journey.data_sources_accessed:
                    data_source_usage[source] += 1
            
            analytics = {
                "total_decision_journeys": total_journeys,
                "average_duration_seconds": avg_duration,
                "average_steps": avg_steps,
                "average_confidence": avg_confidence,
                "data_source_usage": dict(data_source_usage),
                "completion_rate": len(completed_journeys) / len(self.decision_journeys) if self.decision_journeys else 0
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating decision analytics: {e}")
            raise
    
    async def get_feature_adoption_metrics(self) -> Dict[str, Any]:
        """Get feature adoption and usage metrics"""
        try:
            total_usage = sum(self.feature_usage.values())
            
            if total_usage == 0:
                return {"message": "No feature usage data"}
            
            # Calculate adoption rates
            feature_metrics = {}
            for feature, usage_count in self.feature_usage.items():
                adoption_rate = usage_count / total_usage
                feature_metrics[feature] = {
                    "usage_count": usage_count,
                    "adoption_rate": adoption_rate,
                    "popularity_rank": 0  # Will be calculated below
                }
            
            # Rank features by popularity
            sorted_features = sorted(feature_metrics.items(), key=lambda x: x[1]["usage_count"], reverse=True)
            for rank, (feature, metrics) in enumerate(sorted_features, 1):
                feature_metrics[feature]["popularity_rank"] = rank
            
            return {
                "total_feature_interactions": total_usage,
                "unique_features_used": len(self.feature_usage),
                "feature_metrics": feature_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating feature adoption metrics: {e}")
            raise

# Initialize the user analytics engine
analytics_engine = UserBehaviorAnalytics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting User Analytics Service...")
    logger.info("User Analytics Service started successfully")
    yield
    logger.info("User Analytics Service shutting down...")

# Create FastAPI app

# KSE Client for intelligence integration
kse_client = None

async def initialize_kse_client():
    """Initialize KSE client for intelligence integration"""
    global kse_client
    try:
        kse_client = LiftKSEClient()
        print("KSE Client initialized successfully")
        return True
    except Exception as e:
        print(f"KSE Client initialization failed: {e}")
        kse_client = None
        return False

app = FastAPI(
    title="LiftOS User Analytics Service",
    description="Comprehensive user behavior tracking and analysis",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "user-analytics",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/events")
async def track_event(data: Dict[str, Any]):
    """Track a user event"""
    try:
        user_id = data.get('user_id')
        event_type = data.get('event_type')
        event_data = data.get('event_data', {})
        
        if not user_id or not event_type:
            raise HTTPException(status_code=400, detail="user_id and event_type are required")
        
        event = await analytics_engine.track_user_event(user_id, event_type, event_data)
        return {"status": "success", "event_id": event.id}
    except Exception as e:
        logger.error(f"Error tracking event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/start")
async def start_session(data: Dict[str, Any]):
    """Start a user session"""
    try:
        user_id = data.get('user_id')
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        session = await analytics_engine.start_user_session(user_id, data)
        return {"status": "success", "session_id": session.session_id}
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str, data: Dict[str, Any]):
    """End a user session"""
    try:
        session = await analytics_engine.end_user_session(session_id, data)
        return {"status": "success", "session": {
            "session_id": session.session_id,
            "duration": (session.end_time - session.start_time).total_seconds(),
            "events_count": len(session.events),
            "decisions_made": session.decisions_made
        }}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisions/start")
async def start_decision_journey(data: Dict[str, Any]):
    """Start a decision journey"""
    try:
        user_id = data.get('user_id')
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        journey = await analytics_engine.start_decision_journey(user_id, data)
        return {"status": "success", "decision_id": journey.decision_id}
    except Exception as e:
        logger.error(f"Error starting decision journey: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisions/{decision_id}/step")
async def track_decision_step(decision_id: str, data: Dict[str, Any]):
    """Track a decision step"""
    try:
        await analytics_engine.track_decision_step(decision_id, data)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking decision step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisions/{decision_id}/complete")
async def complete_decision_journey(decision_id: str, data: Dict[str, Any]):
    """Complete a decision journey"""
    try:
        journey = await analytics_engine.complete_decision_journey(decision_id, data)
        return {"status": "success", "journey": {
            "decision_id": journey.decision_id,
            "duration": (journey.end_time - journey.start_time).total_seconds(),
            "steps_count": len(journey.steps),
            "confidence_score": journey.confidence_score
        }}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing decision journey: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/users")
async def get_user_analytics(user_id: Optional[str] = None):
    """Get user behavior analytics"""
    try:
        summary = await analytics_engine.get_user_behavior_summary(user_id)
        return {"status": "success", "analytics": summary}
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/decisions")
async def get_decision_analytics():
    """Get decision journey analytics"""
    try:
        analytics = await analytics_engine.get_decision_analytics()
        return {"status": "success", "analytics": analytics}
    except Exception as e:
        logger.error(f"Error getting decision analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/features")
async def get_feature_analytics():
    """Get feature adoption analytics"""
    try:
        metrics = await analytics_engine.get_feature_adoption_metrics()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting feature analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013)