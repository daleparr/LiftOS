"""
Shopify Connector with KSE Universal Substrate Integration
Tier 1 E-commerce Platform Connector for LiftOS v1.3.0
"""
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
import httpx
import json
from pydantic import BaseModel, Field
import uuid

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.causal_marketing import (
    CausalMarketingData, TreatmentType, RandomizationUnit,
    ConfounderVariable, ExternalFactor, CalendarDimension
)
from shared.models.marketing import DataSource, CampaignObjective, AdStatus
from shared.utils.causal_transforms import CausalDataTransformer, TreatmentAssignmentResult
from shared.utils.logging import get_logger
from shared.kse_sdk.client import kse_client
from shared.kse_sdk.causal_models import CausalMemoryEntry, CausalRelationship

logger = get_logger(__name__)


class ShopifyOrderData(BaseModel):
    """Shopify order data model"""
    order_id: str
    order_number: str
    customer_id: Optional[str] = None
    email: Optional[str] = None
    total_price: float
    subtotal_price: float
    total_tax: float
    currency: str
    financial_status: str
    fulfillment_status: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    customer_data: Dict[str, Any] = Field(default_factory=dict)
    shipping_address: Dict[str, Any] = Field(default_factory=dict)
    billing_address: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    source_name: Optional[str] = None
    referring_site: Optional[str] = None
    landing_site: Optional[str] = None
    utm_parameters: Dict[str, str] = Field(default_factory=dict)


class ShopifyProductData(BaseModel):
    """Shopify product data model"""
    product_id: str
    title: str
    handle: str
    product_type: str
    vendor: str
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    status: str
    tags: List[str] = Field(default_factory=list)
    variants: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    options: List[Dict[str, Any]] = Field(default_factory=list)
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    metafields: Dict[str, Any] = Field(default_factory=dict)


class ShopifyCustomerData(BaseModel):
    """Shopify customer data model"""
    customer_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_order_id: Optional[str] = None
    last_order_name: Optional[str] = None
    orders_count: int = 0
    total_spent: float = 0.0
    tags: List[str] = Field(default_factory=list)
    addresses: List[Dict[str, Any]] = Field(default_factory=list)
    accepts_marketing: bool = False
    marketing_opt_in_level: Optional[str] = None
    state: str = "enabled"
    verified_email: bool = False
    tax_exempt: bool = False
    currency: str = "USD"


class ShopifyConnector:
    """Shopify API connector with KSE universal substrate integration"""
    
    def __init__(self, shop_domain: str, access_token: str, api_version: str = "2024-01"):
        self.shop_domain = shop_domain
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = f"https://{shop_domain}.myshopify.com/admin/api/{api_version}"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rate_limit_remaining = 40  # Shopify REST Admin API limit
        self.rate_limit_reset = time.time()
        
        # KSE Integration
        self.causal_transformer = CausalDataTransformer()
        
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Shopify API with rate limiting"""
        await self._handle_rate_limiting()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json"
        }
        
        try:
            response = await self.client.get(url, headers=headers, params=params or {})
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get("X-Shopify-Shop-Api-Call-Limit", "40/40").split("/")[0])
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limited, wait and retry
                await asyncio.sleep(2)
                return await self._make_request(endpoint, params)
            else:
                logger.error(f"Shopify API error: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            logger.error(f"Failed to make Shopify API request: {str(e)}")
            raise
    
    async def _handle_rate_limiting(self):
        """Handle Shopify API rate limiting"""
        if self.rate_limit_remaining <= 2:
            # Wait for rate limit reset (Shopify uses leaky bucket)
            wait_time = max(0, 2 - (time.time() - self.rate_limit_reset))
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            self.rate_limit_remaining = 40
            self.rate_limit_reset = time.time()
    
    async def get_orders(self, date_start: date, date_end: date, limit: int = 250) -> List[ShopifyOrderData]:
        """Get orders data from Shopify API"""
        try:
            orders = []
            params = {
                "created_at_min": date_start.isoformat(),
                "created_at_max": date_end.isoformat(),
                "limit": min(limit, 250),  # Shopify max limit
                "status": "any"
            }
            
            # Handle pagination
            while True:
                data = await self._make_request("orders.json", params)
                batch_orders = data.get("orders", [])
                
                if not batch_orders:
                    break
                
                for order_data in batch_orders:
                    # Extract UTM parameters from referring_site or note
                    utm_params = self._extract_utm_parameters(order_data)
                    
                    order = ShopifyOrderData(
                        order_id=str(order_data["id"]),
                        order_number=order_data["order_number"],
                        customer_id=str(order_data["customer"]["id"]) if order_data.get("customer") else None,
                        email=order_data.get("email"),
                        total_price=float(order_data["total_price"]),
                        subtotal_price=float(order_data["subtotal_price"]),
                        total_tax=float(order_data["total_tax"]),
                        currency=order_data["currency"],
                        financial_status=order_data["financial_status"],
                        fulfillment_status=order_data.get("fulfillment_status"),
                        created_at=datetime.fromisoformat(order_data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(order_data["updated_at"].replace("Z", "+00:00")),
                        line_items=order_data.get("line_items", []),
                        customer_data=order_data.get("customer", {}),
                        shipping_address=order_data.get("shipping_address", {}),
                        billing_address=order_data.get("billing_address", {}),
                        tags=order_data.get("tags", "").split(", ") if order_data.get("tags") else [],
                        note=order_data.get("note"),
                        source_name=order_data.get("source_name"),
                        referring_site=order_data.get("referring_site"),
                        landing_site=order_data.get("landing_site"),
                        utm_parameters=utm_params
                    )
                    orders.append(order)
                
                # Check for next page
                if len(batch_orders) < params["limit"]:
                    break
                
                # Update pagination
                last_order = batch_orders[-1]
                params["since_id"] = last_order["id"]
            
            logger.info(f"Retrieved {len(orders)} orders from Shopify")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get Shopify orders: {str(e)}")
            raise
    
    async def get_products(self, limit: int = 250) -> List[ShopifyProductData]:
        """Get products data from Shopify API"""
        try:
            products = []
            params = {
                "limit": min(limit, 250),
                "published_status": "any"
            }
            
            while True:
                data = await self._make_request("products.json", params)
                batch_products = data.get("products", [])
                
                if not batch_products:
                    break
                
                for product_data in batch_products:
                    product = ShopifyProductData(
                        product_id=str(product_data["id"]),
                        title=product_data["title"],
                        handle=product_data["handle"],
                        product_type=product_data["product_type"],
                        vendor=product_data["vendor"],
                        created_at=datetime.fromisoformat(product_data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(product_data["updated_at"].replace("Z", "+00:00")),
                        published_at=datetime.fromisoformat(product_data["published_at"].replace("Z", "+00:00")) if product_data.get("published_at") else None,
                        status=product_data["status"],
                        tags=product_data.get("tags", "").split(", ") if product_data.get("tags") else [],
                        variants=product_data.get("variants", []),
                        images=product_data.get("images", []),
                        options=product_data.get("options", []),
                        seo_title=product_data.get("seo", {}).get("title"),
                        seo_description=product_data.get("seo", {}).get("description"),
                        metafields=product_data.get("metafields", {})
                    )
                    products.append(product)
                
                if len(batch_products) < params["limit"]:
                    break
                
                last_product = batch_products[-1]
                params["since_id"] = last_product["id"]
            
            logger.info(f"Retrieved {len(products)} products from Shopify")
            return products
            
        except Exception as e:
            logger.error(f"Failed to get Shopify products: {str(e)}")
            raise
    
    async def get_customers(self, date_start: date, date_end: date, limit: int = 250) -> List[ShopifyCustomerData]:
        """Get customers data from Shopify API"""
        try:
            customers = []
            params = {
                "created_at_min": date_start.isoformat(),
                "created_at_max": date_end.isoformat(),
                "limit": min(limit, 250)
            }
            
            while True:
                data = await self._make_request("customers.json", params)
                batch_customers = data.get("customers", [])
                
                if not batch_customers:
                    break
                
                for customer_data in batch_customers:
                    customer = ShopifyCustomerData(
                        customer_id=str(customer_data["id"]),
                        email=customer_data["email"],
                        first_name=customer_data.get("first_name"),
                        last_name=customer_data.get("last_name"),
                        phone=customer_data.get("phone"),
                        created_at=datetime.fromisoformat(customer_data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(customer_data["updated_at"].replace("Z", "+00:00")),
                        last_order_id=str(customer_data["last_order_id"]) if customer_data.get("last_order_id") else None,
                        last_order_name=customer_data.get("last_order_name"),
                        orders_count=customer_data.get("orders_count", 0),
                        total_spent=float(customer_data.get("total_spent", 0)),
                        tags=customer_data.get("tags", "").split(", ") if customer_data.get("tags") else [],
                        addresses=customer_data.get("addresses", []),
                        accepts_marketing=customer_data.get("accepts_marketing", False),
                        marketing_opt_in_level=customer_data.get("marketing_opt_in_level"),
                        state=customer_data.get("state", "enabled"),
                        verified_email=customer_data.get("verified_email", False),
                        tax_exempt=customer_data.get("tax_exempt", False),
                        currency=customer_data.get("currency", "USD")
                    )
                    customers.append(customer)
                
                if len(batch_customers) < params["limit"]:
                    break
                
                last_customer = batch_customers[-1]
                params["since_id"] = last_customer["id"]
            
            logger.info(f"Retrieved {len(customers)} customers from Shopify")
            return customers
            
        except Exception as e:
            logger.error(f"Failed to get Shopify customers: {str(e)}")
            raise
    
    def _extract_utm_parameters(self, order_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract UTM parameters from order data"""
        utm_params = {}
        
        # Check referring_site for UTM parameters
        referring_site = order_data.get("referring_site", "")
        if referring_site and "utm_" in referring_site:
            # Parse UTM parameters from URL
            try:
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(referring_site)
                query_params = parse_qs(parsed.query)
                
                for key, value in query_params.items():
                    if key.startswith("utm_"):
                        utm_params[key] = value[0] if value else ""
            except Exception:
                pass
        
        # Check note for UTM parameters (some apps store them there)
        note = order_data.get("note", "")
        if note and "utm_" in note:
            # Simple extraction from note
            for param in ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"]:
                if param in note:
                    # Extract value after parameter name
                    try:
                        start = note.find(param + "=") + len(param) + 1
                        end = note.find("&", start)
                        if end == -1:
                            end = note.find(" ", start)
                        if end == -1:
                            end = len(note)
                        utm_params[param] = note[start:end].strip()
                    except Exception:
                        pass
        
        return utm_params
    
    async def transform_to_causal_format(
        self, 
        orders: List[ShopifyOrderData], 
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[CausalMarketingData]:
        """Transform Shopify data to causal marketing format with KSE integration"""
        causal_data_list = []
        
        for order in orders:
            try:
                # Create base causal marketing data
                causal_data = await self._create_causal_marketing_data(order, org_id, historical_data)
                
                # Enhance with KSE substrate integration
                await self._enhance_with_kse_substrate(causal_data, order, org_id)
                
                causal_data_list.append(causal_data)
                
            except Exception as e:
                logger.error(f"Failed to transform order {order.order_id} to causal format: {str(e)}")
                continue
        
        logger.info(f"Transformed {len(causal_data_list)} Shopify orders to causal format")
        return causal_data_list
    
    async def _create_causal_marketing_data(
        self, 
        order: ShopifyOrderData, 
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> CausalMarketingData:
        """Create causal marketing data from Shopify order"""
        
        # Determine treatment assignment based on UTM parameters and order characteristics
        treatment_assignment = await self._determine_treatment_assignment(order, historical_data)
        
        # Create calendar features
        calendar_features = await self.causal_transformer._create_calendar_features(order.created_at)
        
        # Detect confounders
        confounders = await self._detect_shopify_confounders(order, historical_data)
        
        # Extract geographic data
        geographic_data = self._extract_geographic_data(order)
        
        # Extract audience data
        audience_data = self._extract_audience_data(order)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(order)
        
        return CausalMarketingData(
            id=f"shopify_order_{order.order_id}_{uuid.uuid4().hex[:8]}",
            org_id=org_id,
            timestamp=order.created_at,
            data_source=DataSource.SHOPIFY,
            campaign_id=order.utm_parameters.get("utm_campaign", f"shopify_organic_{order.source_name}"),
            campaign_name=order.utm_parameters.get("utm_campaign", f"Shopify Organic - {order.source_name}"),
            campaign_objective=self._map_campaign_objective(order.utm_parameters.get("utm_medium")),
            campaign_status=AdStatus.ACTIVE,
            treatment_group=treatment_assignment.treatment_group,
            treatment_type=treatment_assignment.treatment_type,
            treatment_intensity=treatment_assignment.treatment_intensity,
            randomization_unit=treatment_assignment.randomization_unit,
            experiment_id=treatment_assignment.experiment_id,
            spend=0.0,  # Shopify doesn't track ad spend directly
            impressions=0,  # Not available in Shopify
            clicks=1,  # Each order represents at least one "click" to the store
            conversions=1,  # Each order is a conversion
            revenue=order.total_price,
            platform_metrics={
                "order_number": order.order_number,
                "financial_status": order.financial_status,
                "fulfillment_status": order.fulfillment_status,
                "subtotal_price": order.subtotal_price,
                "total_tax": order.total_tax,
                "currency": order.currency,
                "line_items_count": len(order.line_items),
                "customer_orders_count": order.customer_data.get("orders_count", 0),
                "customer_total_spent": order.customer_data.get("total_spent", 0),
                "source_name": order.source_name,
                "referring_site": order.referring_site,
                "landing_site": order.landing_site,
                "tags": order.tags
            },
            confounders=confounders,
            calendar_features=calendar_features,
            geographic_data=geographic_data,
            audience_data=audience_data,
            causal_metadata={
                "platform": "shopify",
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "utm_parameters": order.utm_parameters,
                "data_source_confidence": 0.95  # High confidence for direct e-commerce data
            },
            data_quality_score=data_quality_score
        )
    
    async def _enhance_with_kse_substrate(
        self, 
        causal_data: CausalMarketingData, 
        order: ShopifyOrderData, 
        org_id: str
    ):
        """Enhance causal data with KSE universal substrate integration"""
        try:
            # 1. Neural Embeddings - Create rich text representation
            neural_content = self._create_neural_content(order)
            
            # 2. Conceptual Spaces - Map to conceptual dimensions
            conceptual_space = self._map_to_conceptual_space(order)
            
            # 3. Knowledge Graph - Create relationships
            knowledge_graph_nodes = self._create_knowledge_graph_nodes(order, causal_data)
            
            # Store in KSE with causal relationships
            causal_relationships = [
                CausalRelationship(
                    source_node=f"customer_{order.customer_id}",
                    target_node=f"order_{order.order_id}",
                    relationship_type="purchases",
                    strength=0.95,
                    confidence=0.9,
                    temporal_lag=0,
                    metadata={
                        "order_value": order.total_price,
                        "currency": order.currency,
                        "utm_source": order.utm_parameters.get("utm_source", "direct")
                    }
                ),
                CausalRelationship(
                    source_node=f"campaign_{causal_data.campaign_id}",
                    target_node=f"order_{order.order_id}",
                    relationship_type="drives_conversion",
                    strength=0.8,
                    confidence=0.85,
                    temporal_lag=0,
                    metadata={
                        "attribution_model": "last_click",
                        "revenue": order.total_price
                    }
                )
            ]
            
            # Create causal memory entry
            causal_memory = CausalMemoryEntry(
                content=neural_content,
                causal_relationships=causal_relationships,
                temporal_context={
                    "timestamp": order.created_at.isoformat(),
                    "day_of_week": order.created_at.weekday(),
                    "hour_of_day": order.created_at.hour,
                    "is_weekend": order.created_at.weekday() >= 5
                },
                causal_metadata={
                    "platform": "shopify",
                    "data_type": "e_commerce_order",
                    "treatment_type": causal_data.treatment_type.value,
                    "revenue_impact": order.total_price
                },
                platform_context={
                    "source": "shopify",
                    "order_id": order.order_id,
                    "customer_segment": self._determine_customer_segment(order),
                    "product_categories": self._extract_product_categories(order)
                },
                experiment_id=causal_data.experiment_id
            )
            
            # Store in KSE
            memory_id = await kse_client.store_causal_memory(org_id, causal_memory)
            
            # Update causal data with KSE integration
            causal_data.conceptual_space = conceptual_space
            causal_data.knowledge_graph_nodes = knowledge_graph_nodes
            causal_data.causal_metadata["kse_memory_id"] = memory_id
            causal_data.causal_metadata["neural_embedding_generated"] = True
            causal_data.causal_metadata["conceptual_space_mapped"] = True
            causal_data.causal_metadata["knowledge_graph_integrated"] = True
            
            logger.info(f"Enhanced Shopify order {order.order_id} with KSE substrate integration")
            
        except Exception as e:
            logger.error(f"Failed to enhance order {order.order_id} with KSE substrate: {str(e)}")
            # Continue without KSE enhancement
            causal_data.causal_metadata["kse_integration_failed"] = True
            causal_data.causal_metadata["kse_error"] = str(e)
    
    def _create_neural_content(self, order: ShopifyOrderData) -> str:
        """Create rich text content for neural embedding"""
        content_parts = [
            f"Shopify order {order.order_number} for ${order.total_price} {order.currency}",
            f"Customer: {order.email or 'guest'}",
            f"Source: {order.source_name or 'direct'}",
            f"Financial status: {order.financial_status}",
            f"Items: {len(order.line_items)} products"
        ]
        
        # Add UTM information
        if order.utm_parameters:
            utm_info = ", ".join([f"{k}={v}" for k, v in order.utm_parameters.items() if v])
            if utm_info:
                content_parts.append(f"UTM: {utm_info}")
        
        # Add product information
        if order.line_items:
            product_titles = [item.get("title", "Unknown") for item in order.line_items[:3]]
            content_parts.append(f"Products: {', '.join(product_titles)}")
        
        # Add customer information
        if order.customer_data:
            customer_info = []
            if order.customer_data.get("orders_count"):
                customer_info.append(f"Customer orders: {order.customer_data['orders_count']}")
            if order.customer_data.get("total_spent"):
                customer_info.append(f"Customer LTV: ${order.customer_data['total_spent']}")
            if customer_info:
                content_parts.append(", ".join(customer_info))
        
        return ". ".join(content_parts)
    
    def _map_to_conceptual_space(self, order: ShopifyOrderData) -> str:
        """Map order to conceptual space dimensions"""
        # Determine conceptual space based on order characteristics
        dimensions = []
        
        # Price dimension
        if order.total_price < 50:
            dimensions.append("low_value")
        elif order.total_price < 200:
            dimensions.append("medium_value")
        else:
            dimensions.append("high_value")
        
        # Customer dimension
        customer_orders = order.customer_data.get("orders_count", 0)
        if customer_orders == 0:
            dimensions.append("new_customer")
        elif customer_orders < 5:
            dimensions.append("returning_customer")
        else:
            dimensions.append("loyal_customer")
        
        # Source dimension
        source = order.source_name or "direct"
        if "social" in source.lower():
            dimensions.append("social_commerce")
        elif "search" in source.lower() or order.utm_parameters.get("utm_medium") == "cpc":
            dimensions.append("search_commerce")
        elif order.utm_parameters.get("utm_medium") == "email":
            dimensions.append("email_commerce")
        else:
            dimensions.append("direct_commerce")
        
        return "_".join(dimensions)
    
    def _create_knowledge_graph_nodes(self, order: ShopifyOrderData, causal_data: CausalMarketingData) -> List[str]:
        """Create knowledge graph node connections"""
        nodes = [
            f"order_{order.order_id}",
            f"customer_{order.customer_id}" if order.customer_id else f"guest_{order.email}",
            f"campaign_{causal_data.campaign_id}",
            f"source_{order.source_name}" if order.source_name else "source_direct"
        ]
        
        # Add product nodes
        for item in order.line_items[:5]:  # Limit to first 5 products
            if item.get("product_id"):
                nodes.append(f"product_{item['product_id']}")
        
        # Add UTM nodes
        if order.utm_parameters.get("utm_source"):
            nodes.append(f"utm_source_{order.utm_parameters['utm_source']}")
        if order.utm_parameters.get("utm_medium"):
            nodes.append(f"utm_medium_{order.utm_parameters['utm_medium']}")
        
        return nodes
    
    async def _determine_treatment_assignment(
        self, 
        order: ShopifyOrderData, 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for Shopify order"""
        
        # Analyze UTM parameters for treatment identification
        utm_source = order.utm_parameters.get("utm_source", "")
        utm_medium = order.utm_parameters.get("utm_medium", "")
        utm_campaign = order.utm_parameters.get("utm_campaign", "")
        
        # Determine treatment type
        if utm_medium in ["cpc", "ppc", "paid"]:
            treatment_type = TreatmentType.BUDGET_INCREASE
            treatment_group = "paid_acquisition"
            treatment_intensity = 0.8
        elif utm_medium == "email":
            treatment_type = TreatmentType.TARGETING_CHANGE
            treatment_group = "email_marketing"
            treatment_intensity = 0.6
        elif utm_medium in ["social", "facebook", "instagram", "twitter"]:
            treatment_type = TreatmentType.CREATIVE_CHANGE
            treatment_group = "social_media"
            treatment_intensity = 0.7
        elif utm_source or order.referring_site:
            treatment_type = TreatmentType.TARGETING_CHANGE
            treatment_group = "referral_traffic"
            treatment_intensity = 0.5
        else:
            treatment_type = TreatmentType.CONTROL
            treatment_group = "organic_direct"
            treatment_intensity = 0.0
        
        # Determine randomization unit
        if order.customer_id:
            randomization_unit = RandomizationUnit.USER_SEGMENT
        else:
            randomization_unit = RandomizationUnit.TIME_PERIOD
        
        # Generate experiment ID if this appears to be part of a campaign
        experiment_id = None
        if utm_campaign and utm_campaign != "":
            experiment_id = f"shopify_campaign_{utm_campaign}_{order.created_at.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.8
        )
    
    async def _detect_shopify_confounders(
        self, 
        order: ShopifyOrderData, 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect Shopify-specific confounding variables"""
        confounders = []
        
        # Customer lifetime value confounder
        customer_ltv = order.customer_data.get("total_spent", 0)
        if customer_ltv > 0:
            confounders.append(ConfounderVariable(
                variable_name="customer_lifetime_value",
                variable_type="continuous",
                value=customer_ltv,
                importance_score=0.85,
                detection_method="customer_data_analysis",
                control_strategy="include_customer_ltv_as_covariate"
            ))
        
        # Repeat customer confounder
        customer_orders = order.customer_data.get("orders_count", 0)
        if customer_orders > 1:
            confounders.append(ConfounderVariable(
                variable_name="repeat_customer",
                variable_type="binary",
                value=True,
                importance_score=0.75,
                detection_method="customer_order_history",
                control_strategy="include_customer_type_indicator"
            ))
        
        # Cart size confounder
        cart_size = len(order.line_items)
        if cart_size > 3:
            confounders.append(ConfounderVariable(
                variable_name="large_cart_size",
                variable_type="continuous",
                value=cart_size,
                importance_score=0.7,
                detection_method="cart_analysis",
                control_strategy="include_cart_size_as_covariate"
            ))
        
        # Seasonal/holiday confounder
        if self._is_holiday_period(order.created_at):
            confounders.append(ConfounderVariable(
                variable_name="holiday_period",
                variable_type="binary",
                value=True,
                importance_score=0.8,
                detection_method="calendar_analysis",
                control_strategy="include_holiday_indicator"
            ))
        
        return confounders
    
    def _extract_geographic_data(self, order: ShopifyOrderData) -> Dict[str, Any]:
        """Extract geographic information from order"""
        geographic_data = {}
        
        # Shipping address
        if order.shipping_address:
            geographic_data.update({
                "shipping_country": order.shipping_address.get("country"),
                "shipping_province": order.shipping_address.get("province"),
                "shipping_city": order.shipping_address.get("city"),
                "shipping_zip": order.shipping_address.get("zip")
            })
        
        # Billing address
        if order.billing_address:
            geographic_data.update({
                "billing_country": order.billing_address.get("country"),
                "billing_province": order.billing_address.get("province"),
                "billing_city": order.billing_address.get("city"),
                "billing_zip": order.billing_address.get("zip")
            })
        
        return geographic_data
    
    def _extract_audience_data(self, order: ShopifyOrderData) -> Dict[str, Any]:
        """Extract audience characteristics from order"""
        audience_data = {}
        
        # Customer data
        if order.customer_data:
            audience_data.update({
                "customer_type": "returning" if order.customer_data.get("orders_count", 0) > 0 else "new",
                "customer_orders_count": order.customer_data.get("orders_count", 0),
                "customer_total_spent": order.customer_data.get("total_spent", 0),
                "accepts_marketing": order.customer_data.get("accepts_marketing", False)
            })
        
        # Order characteristics
        audience_data.update({
            "order_value_segment": self._categorize_order_value(order.total_price),
            "cart_size": len(order.line_items),
            "has_discount": any("discount" in str(item).lower() for item in order.line_items),
            "product_categories": self._extract_product_categories(order)
        })
        
        return audience_data
    
    def _calculate_data_quality_score(self, order: ShopifyOrderData) -> float:
        """Calculate data quality score for causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [order.order_id, order.total_price, order.created_at, order.currency]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Customer data quality
        customer_score = 0.5  # Base score
        if order.customer_id:
            customer_score += 0.3
        if order.email:
            customer_score += 0.2
        score_components.append(min(customer_score, 1.0) * 0.2)
        
        # Attribution data quality
        attribution_score = 0.3  # Base score for having order data
        if order.utm_parameters:
            attribution_score += 0.4
        if order.referring_site:
            attribution_score += 0.2
        if order.source_name:
            attribution_score += 0.1
        score_components.append(min(attribution_score, 1.0) * 0.3)
        
        # Product data quality
        product_score = 0.5 if order.line_items else 0.0
        if order.line_items:
            product_score += 0.3 if len(order.line_items) > 0 else 0.0
            product_score += 0.2 if any(item.get("product_id") for item in order.line_items) else 0.0
        score_components.append(min(product_score, 1.0) * 0.2)
        
        return sum(score_components)
    
    def _map_campaign_objective(self, utm_medium: Optional[str]) -> CampaignObjective:
        """Map UTM medium to campaign objective"""
        if not utm_medium:
            return CampaignObjective.CONVERSIONS
        
        utm_medium = utm_medium.lower()
        if utm_medium in ["cpc", "ppc", "paid"]:
            return CampaignObjective.CONVERSIONS
        elif utm_medium == "email":
            return CampaignObjective.ENGAGEMENT
        elif utm_medium in ["social", "facebook", "instagram"]:
            return CampaignObjective.BRAND_AWARENESS
        else:
            return CampaignObjective.CONVERSIONS
    
    def _determine_customer_segment(self, order: ShopifyOrderData) -> str:
        """Determine customer segment based on order data"""
        customer_orders = order.customer_data.get("orders_count", 0)
        customer_ltv = order.customer_data.get("total_spent", 0)
        
        if customer_orders == 0:
            return "new_customer"
        elif customer_orders < 3:
            return "occasional_customer"
        elif customer_ltv > 500:
            return "high_value_customer"
        else:
            return "regular_customer"
    
    def _extract_product_categories(self, order: ShopifyOrderData) -> List[str]:
        """Extract product categories from line items"""
        categories = set()
        
        for item in order.line_items:
            # Extract from product type or title
            if item.get("product_type"):
                categories.add(item["product_type"].lower())
            
            # Extract from title keywords
            title = item.get("title", "").lower()
            if "clothing" in title or "apparel" in title:
                categories.add("clothing")
            elif "electronics" in title or "tech" in title:
                categories.add("electronics")
            elif "home" in title or "furniture" in title:
                categories.add("home_goods")
            elif "beauty" in title or "cosmetic" in title:
                categories.add("beauty")
            elif "book" in title:
                categories.add("books")
        
        return list(categories)
    
    def _categorize_order_value(self, total_price: float) -> str:
        """Categorize order value into segments"""
        if total_price < 25:
            return "low_value"
        elif total_price < 100:
            return "medium_value"
        elif total_price < 300:
            return "high_value"
        else:
            return "premium_value"
    
    def _is_holiday_period(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within major holiday periods"""
        month = timestamp.month
        day = timestamp.day
        
        # Major shopping holidays
        holiday_periods = [
            (11, 20, 11, 30),  # Black Friday/Cyber Monday
            (12, 15, 12, 31),  # Christmas shopping
            (1, 1, 1, 15),     # New Year sales
            (2, 10, 2, 20),    # Valentine's Day
            (5, 1, 5, 15),     # Mother's Day
            (6, 10, 6, 20),    # Father's Day
            (7, 1, 7, 10),     # Independence Day sales
        ]
        
        for start_month, start_day, end_month, end_day in holiday_periods:
            if (month == start_month and day >= start_day) or \
               (month == end_month and day <= end_day) or \
               (start_month < month < end_month):
                return True
        
        return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

