"""
KSE Memory SDK Models
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel


class MemorySearchResult(BaseModel):
    """Result from memory search operations"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    memory_type: str
    organization_id: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryInsights(BaseModel):
    """Analytics and insights about memory usage"""
    total_memories: int
    memory_types: Dict[str, int]
    organizations: Dict[str, int]
    recent_activity: List[Dict[str, Any]]
    storage_stats: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryEntry(BaseModel):
    """Memory entry for storage"""
    content: str
    memory_type: str
    organization_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str
    memory_type: Optional[str] = None
    organization_id: str
    limit: int = 10
    threshold: float = 0.7
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }