"""
Billing Service for Lift OS Core

Handles subscription management, usage tracking, and Stripe integration.
Provides billing analytics and cost management for organizations.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal

import stripe
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import shared utilities
import sys
sys.path.append('/app/shared')

from models.base import User, Organization, BillingPlan, UsageRecord, Invoice
from auth.jwt_utils import verify_token, require_permissions
from utils.logging import get_logger, log_request
from utils.config import get_config

# Initialize logging
logger = get_logger(__name__)
config = get_config()

# Initialize Stripe
stripe.api_key = config.stripe_secret_key

app = FastAPI(
    title="Lift OS Billing Service",
    description="Billing and subscription management for Lift OS Core",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
billing_plans: Dict[str, BillingPlan] = {}
usage_records: Dict[str, List[UsageRecord]] = {}
invoices: Dict[str, List[Invoice]] = {}
subscriptions: Dict[str, Dict[str, Any]] = {}

# Initialize default billing plans
def initialize_default_plans():
    """Initialize default billing plans"""
    default_plans = [
        BillingPlan(
            plan_id="starter",
            name="Starter Plan",
            description="Perfect for small teams getting started",
            price_monthly=29.00,
            price_yearly=290.00,
            features={
                "max_users": 5,
                "max_modules": 3,
                "memory_storage_gb": 10,
                "api_calls_monthly": 10000,
                "support_level": "community"
            },
            limits={
                "memory_queries_daily": 1000,
                "module_deployments_monthly": 10,
                "data_export_monthly": 5
            }
        ),
        BillingPlan(
            plan_id="professional",
            name="Professional Plan",
            description="For growing teams with advanced needs",
            price_monthly=99.00,
            price_yearly=990.00,
            features={
                "max_users": 25,
                "max_modules": 10,
                "memory_storage_gb": 100,
                "api_calls_monthly": 100000,
                "support_level": "email"
            },
            limits={
                "memory_queries_daily": 10000,
                "module_deployments_monthly": 50,
                "data_export_monthly": 25
            }
        ),
        BillingPlan(
            plan_id="enterprise",
            name="Enterprise Plan",
            description="For large organizations with custom requirements",
            price_monthly=299.00,
            price_yearly=2990.00,
            features={
                "max_users": -1,  # Unlimited
                "max_modules": -1,  # Unlimited
                "memory_storage_gb": 1000,
                "api_calls_monthly": 1000000,
                "support_level": "priority"
            },
            limits={
                "memory_queries_daily": 100000,
                "module_deployments_monthly": 200,
                "data_export_monthly": 100
            }
        )
    ]
    
    for plan in default_plans:
        billing_plans[plan.plan_id] = plan

# Request/Response Models
class CreateSubscriptionRequest(BaseModel):
    organization_id: str
    plan_id: str
    billing_cycle: str = Field(..., regex="^(monthly|yearly)$")
    payment_method_id: Optional[str] = None

class UpdateSubscriptionRequest(BaseModel):
    plan_id: Optional[str] = None
    billing_cycle: Optional[str] = Field(None, regex="^(monthly|yearly)$")

class RecordUsageRequest(BaseModel):
    organization_id: str
    service_name: str
    usage_type: str
    quantity: int
    metadata: Optional[Dict[str, Any]] = None

class BillingAnalyticsResponse(BaseModel):
    organization_id: str
    current_period_usage: Dict[str, Any]
    billing_summary: Dict[str, Any]
    cost_breakdown: Dict[str, Decimal]
    usage_trends: List[Dict[str, Any]]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the billing service"""
    logger.info("Starting Billing Service")
    initialize_default_plans()
    logger.info(f"Initialized {len(billing_plans)} billing plans")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "billing",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Billing Plans
@app.get("/api/v1/plans", response_model=List[BillingPlan])
async def get_billing_plans():
    """Get all available billing plans"""
    logger.info("Fetching billing plans")
    return list(billing_plans.values())

@app.get("/api/v1/plans/{plan_id}", response_model=BillingPlan)
async def get_billing_plan(plan_id: str):
    """Get a specific billing plan"""
    if plan_id not in billing_plans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Billing plan {plan_id} not found"
        )
    
    return billing_plans[plan_id]

# Subscriptions
@app.post("/api/v1/subscriptions")
async def create_subscription(
    request: CreateSubscriptionRequest,
    current_user: User = Depends(verify_token)
):
    """Create a new subscription for an organization"""
    logger.info(f"Creating subscription for org {request.organization_id}")
    
    # Verify user has billing permissions for the organization
    if not require_permissions(current_user, ["billing:write"], request.organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for billing operations"
        )
    
    # Validate plan exists
    if request.plan_id not in billing_plans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Billing plan {request.plan_id} not found"
        )
    
    plan = billing_plans[request.plan_id]
    
    try:
        # Create Stripe subscription
        price = plan.price_yearly if request.billing_cycle == "yearly" else plan.price_monthly
        
        stripe_subscription = stripe.Subscription.create(
            customer=f"org_{request.organization_id}",  # Assume customer exists
            items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': plan.name,
                    },
                    'unit_amount': int(price * 100),  # Convert to cents
                    'recurring': {
                        'interval': 'year' if request.billing_cycle == "yearly" else 'month',
                    },
                },
            }],
            payment_behavior='default_incomplete',
            expand=['latest_invoice.payment_intent'],
        )
        
        # Store subscription locally
        subscription_data = {
            "subscription_id": stripe_subscription.id,
            "organization_id": request.organization_id,
            "plan_id": request.plan_id,
            "billing_cycle": request.billing_cycle,
            "status": stripe_subscription.status,
            "current_period_start": datetime.fromtimestamp(stripe_subscription.current_period_start),
            "current_period_end": datetime.fromtimestamp(stripe_subscription.current_period_end),
            "created_at": datetime.utcnow(),
            "stripe_data": stripe_subscription
        }
        
        subscriptions[request.organization_id] = subscription_data
        
        logger.info(f"Created subscription {stripe_subscription.id} for org {request.organization_id}")
        
        return {
            "subscription_id": stripe_subscription.id,
            "status": stripe_subscription.status,
            "client_secret": stripe_subscription.latest_invoice.payment_intent.client_secret
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment processing error: {str(e)}"
        )

@app.get("/api/v1/subscriptions/{organization_id}")
async def get_subscription(
    organization_id: str,
    current_user: User = Depends(verify_token)
):
    """Get subscription details for an organization"""
    
    # Verify user has billing permissions for the organization
    if not require_permissions(current_user, ["billing:read"], organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for billing operations"
        )
    
    if organization_id not in subscriptions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No subscription found for organization {organization_id}"
        )
    
    subscription = subscriptions[organization_id]
    
    # Remove sensitive Stripe data from response
    response_data = {k: v for k, v in subscription.items() if k != "stripe_data"}
    response_data["plan"] = billing_plans[subscription["plan_id"]]
    
    return response_data

@app.put("/api/v1/subscriptions/{organization_id}")
async def update_subscription(
    organization_id: str,
    request: UpdateSubscriptionRequest,
    current_user: User = Depends(verify_token)
):
    """Update an existing subscription"""
    
    # Verify user has billing permissions for the organization
    if not require_permissions(current_user, ["billing:write"], organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for billing operations"
        )
    
    if organization_id not in subscriptions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No subscription found for organization {organization_id}"
        )
    
    subscription = subscriptions[organization_id]
    
    try:
        # Update Stripe subscription if needed
        update_data = {}
        
        if request.plan_id and request.plan_id != subscription["plan_id"]:
            if request.plan_id not in billing_plans:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Billing plan {request.plan_id} not found"
                )
            
            plan = billing_plans[request.plan_id]
            billing_cycle = request.billing_cycle or subscription["billing_cycle"]
            price = plan.price_yearly if billing_cycle == "yearly" else plan.price_monthly
            
            # Update subscription items
            stripe.Subscription.modify(
                subscription["subscription_id"],
                items=[{
                    'id': subscription["stripe_data"].items.data[0].id,
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': plan.name,
                        },
                        'unit_amount': int(price * 100),
                        'recurring': {
                            'interval': 'year' if billing_cycle == "yearly" else 'month',
                        },
                    },
                }]
            )
            
            subscription["plan_id"] = request.plan_id
            subscription["billing_cycle"] = billing_cycle
        
        logger.info(f"Updated subscription for org {organization_id}")
        
        return {"status": "updated", "subscription_id": subscription["subscription_id"]}
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error updating subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment processing error: {str(e)}"
        )

# Usage Tracking
@app.post("/api/v1/usage")
async def record_usage(
    request: RecordUsageRequest,
    current_user: User = Depends(verify_token)
):
    """Record usage for billing purposes"""
    
    # Verify user has usage recording permissions
    if not require_permissions(current_user, ["usage:write"], request.organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for usage recording"
        )
    
    usage_record = UsageRecord(
        organization_id=request.organization_id,
        service_name=request.service_name,
        usage_type=request.usage_type,
        quantity=request.quantity,
        timestamp=datetime.utcnow(),
        metadata=request.metadata or {}
    )
    
    if request.organization_id not in usage_records:
        usage_records[request.organization_id] = []
    
    usage_records[request.organization_id].append(usage_record)
    
    logger.info(f"Recorded usage: {request.service_name}.{request.usage_type} = {request.quantity} for org {request.organization_id}")
    
    return {"status": "recorded", "usage_id": usage_record.usage_id}

@app.get("/api/v1/usage/{organization_id}")
async def get_usage_summary(
    organization_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(verify_token)
):
    """Get usage summary for an organization"""
    
    # Verify user has usage read permissions
    if not require_permissions(current_user, ["usage:read"], organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for usage data"
        )
    
    if organization_id not in usage_records:
        return {"organization_id": organization_id, "usage": [], "summary": {}}
    
    org_usage = usage_records[organization_id]
    
    # Filter by date range if provided
    if start_date or end_date:
        filtered_usage = []
        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
        
        for record in org_usage:
            if start_dt <= record.timestamp <= end_dt:
                filtered_usage.append(record)
        
        org_usage = filtered_usage
    
    # Calculate summary
    summary = {}
    for record in org_usage:
        key = f"{record.service_name}.{record.usage_type}"
        if key not in summary:
            summary[key] = 0
        summary[key] += record.quantity
    
    return {
        "organization_id": organization_id,
        "usage": [record.dict() for record in org_usage],
        "summary": summary,
        "period": {
            "start": start_date,
            "end": end_date
        }
    }

# Analytics
@app.get("/api/v1/analytics/{organization_id}", response_model=BillingAnalyticsResponse)
async def get_billing_analytics(
    organization_id: str,
    current_user: User = Depends(verify_token)
):
    """Get comprehensive billing analytics for an organization"""
    
    # Verify user has analytics permissions
    if not require_permissions(current_user, ["analytics:read"], organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for analytics data"
        )
    
    # Get current subscription
    subscription = subscriptions.get(organization_id)
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No subscription found for organization {organization_id}"
        )
    
    plan = billing_plans[subscription["plan_id"]]
    
    # Calculate current period usage
    current_period_start = subscription["current_period_start"]
    current_period_end = subscription["current_period_end"]
    
    current_usage = []
    if organization_id in usage_records:
        for record in usage_records[organization_id]:
            if current_period_start <= record.timestamp <= current_period_end:
                current_usage.append(record)
    
    # Usage summary
    usage_summary = {}
    for record in current_usage:
        key = f"{record.service_name}.{record.usage_type}"
        if key not in usage_summary:
            usage_summary[key] = 0
        usage_summary[key] += record.quantity
    
    # Cost breakdown (simplified)
    base_cost = plan.price_yearly if subscription["billing_cycle"] == "yearly" else plan.price_monthly
    overage_costs = Decimal("0.00")  # Calculate based on usage limits
    
    cost_breakdown = {
        "base_subscription": Decimal(str(base_cost)),
        "usage_overages": overage_costs,
        "total": Decimal(str(base_cost)) + overage_costs
    }
    
    # Usage trends (last 6 periods)
    trends = []
    for i in range(6):
        period_start = current_period_start - timedelta(days=30 * (i + 1))
        period_end = current_period_start - timedelta(days=30 * i)
        
        period_usage = 0
        if organization_id in usage_records:
            for record in usage_records[organization_id]:
                if period_start <= record.timestamp <= period_end:
                    period_usage += record.quantity
        
        trends.append({
            "period": period_start.strftime("%Y-%m"),
            "total_usage": period_usage
        })
    
    return BillingAnalyticsResponse(
        organization_id=organization_id,
        current_period_usage=usage_summary,
        billing_summary={
            "plan": plan.name,
            "billing_cycle": subscription["billing_cycle"],
            "next_billing_date": subscription["current_period_end"].isoformat(),
            "status": subscription["status"]
        },
        cost_breakdown=cost_breakdown,
        usage_trends=list(reversed(trends))
    )

# Webhooks
@app.post("/api/v1/webhooks/stripe")
async def stripe_webhook(request):
    """Handle Stripe webhooks for subscription events"""
    # This would handle Stripe webhook events in production
    # For now, just log the event
    logger.info("Received Stripe webhook")
    return {"status": "received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)