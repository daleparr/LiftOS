"""
LiftOS Intelligence SDK
Easy integration with intelligence services for learning and decision-making
"""
from .client import IntelligenceClient
from .learning import LearningManager
from .decision import DecisionManager
from .feedback import FeedbackManager

__all__ = [
    "IntelligenceClient",
    "LearningManager", 
    "DecisionManager",
    "FeedbackManager"
]