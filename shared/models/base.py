"""
Base models and schemas for Lift OS Core
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class UserRole(str, Enum):
    """User roles in the system"""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class SubscriptionTier(str, Enum):
    """Subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ModuleStatus(str, Enum):
    """Module status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class User(TimestampMixin):
    """User model"""
    id: str
    email: str
    name: str
    org_id: str
    roles: List[UserRole] = [UserRole.USER]
    is_active: bool = True
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class Organization(TimestampMixin):
    """Organization model"""
    id: str
    name: str
    domain: Optional[str] = None
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    settings: Dict[str, Any] = {}
    is_active: bool = True


class Module(TimestampMixin):
    """Module registration model"""
    id: str
    name: str
    version: str
    base_url: str
    health_endpoint: str = "/health"
    api_prefix: str = "/api/v1"
    status: ModuleStatus = ModuleStatus.ACTIVE
    features: List[str] = []
    permissions: List[str] = []
    memory_requirements: Dict[str, Any] = {}
    ui_components: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class MemoryContext(TimestampMixin):
    """Memory context model"""
    id: str
    org_id: str
    context_type: str = "general"
    domain: str = "general"
    settings: Dict[str, Any] = {}
    is_active: bool = True


class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    dependencies: Dict[str, str] = {}
    uptime: Optional[float] = None


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1)
    size: int = Field(10, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field("asc", pattern="^(asc|desc)$")


class PaginatedResponse(BaseModel):
    """Paginated response format"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


class JWTClaims(BaseModel):
    """JWT token claims"""
    sub: str  # user_id
    org_id: str
    email: str
    roles: List[UserRole]
    permissions: List[str]
    memory_context: Optional[str] = None
    subscription_tier: SubscriptionTier
    exp: int
    iat: int


class MemorySearchRequest(BaseModel):
    """Memory search request"""
    query: str
    search_type: str = Field("hybrid", pattern="^(neural|conceptual|knowledge|hybrid)$")
    limit: int = Field(10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    memory_type: Optional[str] = None


class MemorySearchResult(BaseModel):
    """Memory search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    memory_type: str
    timestamp: datetime


class MemoryStoreRequest(BaseModel):
    """Memory store request"""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    memory_type: str = "general"


class MemoryInsights(BaseModel):
    """Memory analytics and insights"""
    total_memories: int
    dominant_concepts: List[str]
    knowledge_density: float
    temporal_patterns: Dict[str, Any]
    semantic_clusters: List[Dict[str, Any]]
    memory_types: Dict[str, int]


class BillingUsage(TimestampMixin):
    """Billing usage tracking"""
    id: str
    org_id: str
    user_id: str
    service: str
    operation: str
    quantity: int = 1
    cost: float = 0.0
    metadata: Dict[str, Any] = {}


class Subscription(TimestampMixin):
    """Subscription model"""
    id: str
    org_id: str
    tier: SubscriptionTier
    stripe_subscription_id: Optional[str] = None
    status: str = "active"
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    metadata: Dict[str, Any] = {}


class BillingPlan(BaseModel):
    """Billing plan model"""
    plan_id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: Dict[str, Any] = {}
    limits: Dict[str, Any] = {}
    is_active: bool = True


class UsageRecord(TimestampMixin):
    """Usage record for billing"""
    usage_id: str = Field(default_factory=lambda: f"usage_{int(datetime.utcnow().timestamp())}")
    organization_id: str
    service_name: str
    usage_type: str
    quantity: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class Invoice(TimestampMixin):
    """Invoice model"""
    invoice_id: str
    organization_id: str
    subscription_id: str
    amount: float
    currency: str = "usd"
    status: str = "pending"
    due_date: datetime
    paid_at: Optional[datetime] = None
    stripe_invoice_id: Optional[str] = None
    line_items: List[Dict[str, Any]] = []