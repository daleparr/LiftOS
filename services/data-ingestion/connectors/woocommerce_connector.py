"""
WooCommerce Connector with WordPress Content Integration and KSE Universal Substrate
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
import base64

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


class WooCommerceOrderData(BaseModel):
    """WooCommerce order data model"""
    order_id: str
    order_key: str
    order_number: str
    status: str
    currency: str
    date_created: datetime
    date_modified: datetime
    discount_total: float
    discount_tax: float
    shipping_total: float
    shipping_tax: float
    cart_tax: float
    total: float
    total_tax: float
    customer_id: Optional[str] = None
    customer_ip_address: Optional[str] = None
    customer_user_agent: Optional[str] = None
    customer_note: Optional[str] = None
    billing: Dict[str, Any] = Field(default_factory=dict)
    shipping: Dict[str, Any] = Field(default_factory=dict)
    payment_method: Optional[str] = None
    payment_method_title: Optional[str] = None
    transaction_id: Optional[str] = None
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    tax_lines: List[Dict[str, Any]] = Field(default_factory=list)
    shipping_lines: List[Dict[str, Any]] = Field(default_factory=list)
    fee_lines: List[Dict[str, Any]] = Field(default_factory=list)
    coupon_lines: List[Dict[str, Any]] = Field(default_factory=list)
    refunds: List[Dict[str, Any]] = Field(default_factory=list)
    meta_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    # WordPress/Content Marketing Integration
    referring_post_id: Optional[str] = None
    referring_post_title: Optional[str] = None
    referring_post_type: Optional[str] = None
    utm_parameters: Dict[str, str] = Field(default_factory=dict)
    content_attribution: Dict[str, Any] = Field(default_factory=dict)


class WooCommerceProductData(BaseModel):
    """WooCommerce product data model"""
    product_id: str
    name: str
    slug: str
    permalink: str
    date_created: datetime
    date_modified: datetime
    type: str  # simple, grouped, external, variable
    status: str  # draft, pending, private, publish
    featured: bool
    catalog_visibility: str
    description: str
    short_description: str
    sku: Optional[str] = None
    price: str
    regular_price: str
    sale_price: str
    date_on_sale_from: Optional[datetime] = None
    date_on_sale_to: Optional[datetime] = None
    price_html: str
    on_sale: bool
    purchasable: bool
    total_sales: int
    virtual: bool
    downloadable: bool
    downloads: List[Dict[str, Any]] = Field(default_factory=list)
    download_limit: int
    download_expiry: int
    external_url: str
    button_text: str
    tax_status: str
    tax_class: str
    manage_stock: bool
    stock_quantity: Optional[int] = None
    stock_status: str
    backorders: str
    backorders_allowed: bool
    backordered: bool
    weight: str
    dimensions: Dict[str, str] = Field(default_factory=dict)
    shipping_required: bool
    shipping_taxable: bool
    shipping_class: str
    shipping_class_id: int
    reviews_allowed: bool
    average_rating: str
    rating_count: int
    related_ids: List[int] = Field(default_factory=list)
    upsell_ids: List[int] = Field(default_factory=list)
    cross_sell_ids: List[int] = Field(default_factory=list)
    parent_id: int
    purchase_note: str
    categories: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    attributes: List[Dict[str, Any]] = Field(default_factory=list)
    default_attributes: List[Dict[str, Any]] = Field(default_factory=list)
    variations: List[int] = Field(default_factory=list)
    grouped_products: List[int] = Field(default_factory=list)
    menu_order: int
    meta_data: List[Dict[str, Any]] = Field(default_factory=list)


class WordPressPostData(BaseModel):
    """WordPress post data for content attribution"""
    post_id: str
    title: str
    content: str
    excerpt: str
    status: str
    type: str
    slug: str
    permalink: str
    date: datetime
    date_gmt: datetime
    modified: datetime
    modified_gmt: datetime
    author: int
    featured_media: int
    comment_status: str
    ping_status: str
    sticky: bool
    template: str
    format: str
    categories: List[int] = Field(default_factory=list)
    tags: List[int] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    
    # SEO and Analytics
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    page_views: int = 0
    social_shares: Dict[str, int] = Field(default_factory=dict)
    conversion_rate: float = 0.0


class WooCommerceConnector:
    """WooCommerce API connector with WordPress content integration and KSE substrate"""
    
    def __init__(self, site_url: str, consumer_key: str, consumer_secret: str, wp_username: str = None, wp_password: str = None):
        self.site_url = site_url.rstrip('/')
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.wp_username = wp_username
        self.wp_password = wp_password
        
        # API endpoints
        self.wc_api_url = f"{self.site_url}/wp-json/wc/v3"
        self.wp_api_url = f"{self.site_url}/wp-json/wp/v2"
        
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rate_limit_remaining = 100  # WooCommerce is more generous
        self.rate_limit_reset = time.time()
        
        # KSE Integration
        self.causal_transformer = CausalDataTransformer()
        
        # WordPress authentication
        self.wp_auth = None
        if wp_username and wp_password:
            credentials = base64.b64encode(f"{wp_username}:{wp_password}".encode()).decode()
            self.wp_auth = f"Basic {credentials}"
    
    async def _make_wc_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to WooCommerce API"""
        await self._handle_rate_limiting()
        
        url = f"{self.wc_api_url}/{endpoint}"
        auth = (self.consumer_key, self.consumer_secret)
        
        try:
            response = await self.client.get(url, auth=auth, params=params or {})
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(2)
                return await self._make_wc_request(endpoint, params)
            else:
                logger.error(f"WooCommerce API error: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            logger.error(f"Failed to make WooCommerce API request: {str(e)}")
            raise
    
    async def _make_wp_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to WordPress API"""
        url = f"{self.wp_api_url}/{endpoint}"
        headers = {}
        
        if self.wp_auth:
            headers["Authorization"] = self.wp_auth
        
        try:
            response = await self.client.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to make WordPress API request: {str(e)}")
            raise
    
    async def _handle_rate_limiting(self):
        """Handle API rate limiting"""
        if self.rate_limit_remaining <= 5:
            wait_time = max(0, 1 - (time.time() - self.rate_limit_reset))
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            self.rate_limit_remaining = 100
            self.rate_limit_reset = time.time()
    
    async def get_orders(self, date_start: date, date_end: date, limit: int = 100) -> List[WooCommerceOrderData]:
        """Get orders data from WooCommerce API"""
        try:
            orders = []
            page = 1
            per_page = min(limit, 100)
            
            while len(orders) < limit:
                params = {
                    "after": date_start.isoformat(),
                    "before": date_end.isoformat(),
                    "per_page": per_page,
                    "page": page,
                    "status": "any"
                }
                
                data = await self._make_wc_request("orders", params)
                
                if not data:
                    break
                
                for order_data in data:
                    # Extract UTM parameters and content attribution
                    utm_params, content_attribution = await self._extract_attribution_data(order_data)
                    
                    order = WooCommerceOrderData(
                        order_id=str(order_data["id"]),
                        order_key=order_data["order_key"],
                        order_number=order_data["number"],
                        status=order_data["status"],
                        currency=order_data["currency"],
                        date_created=datetime.fromisoformat(order_data["date_created"].replace("Z", "+00:00")),
                        date_modified=datetime.fromisoformat(order_data["date_modified"].replace("Z", "+00:00")),
                        discount_total=float(order_data["discount_total"]),
                        discount_tax=float(order_data["discount_tax"]),
                        shipping_total=float(order_data["shipping_total"]),
                        shipping_tax=float(order_data["shipping_tax"]),
                        cart_tax=float(order_data["cart_tax"]),
                        total=float(order_data["total"]),
                        total_tax=float(order_data["total_tax"]),
                        customer_id=str(order_data["customer_id"]) if order_data["customer_id"] else None,
                        customer_ip_address=order_data.get("customer_ip_address"),
                        customer_user_agent=order_data.get("customer_user_agent"),
                        customer_note=order_data.get("customer_note"),
                        billing=order_data.get("billing", {}),
                        shipping=order_data.get("shipping", {}),
                        payment_method=order_data.get("payment_method"),
                        payment_method_title=order_data.get("payment_method_title"),
                        transaction_id=order_data.get("transaction_id"),
                        line_items=order_data.get("line_items", []),
                        tax_lines=order_data.get("tax_lines", []),
                        shipping_lines=order_data.get("shipping_lines", []),
                        fee_lines=order_data.get("fee_lines", []),
                        coupon_lines=order_data.get("coupon_lines", []),
                        refunds=order_data.get("refunds", []),
                        meta_data=order_data.get("meta_data", []),
                        utm_parameters=utm_params,
                        content_attribution=content_attribution
                    )
                    orders.append(order)
                
                if len(data) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Retrieved {len(orders)} orders from WooCommerce")
            return orders[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get WooCommerce orders: {str(e)}")
            raise
    
    async def get_products(self, limit: int = 100) -> List[WooCommerceProductData]:
        """Get products data from WooCommerce API"""
        try:
            products = []
            page = 1
            per_page = min(limit, 100)
            
            while len(products) < limit:
                params = {
                    "per_page": per_page,
                    "page": page,
                    "status": "any"
                }
                
                data = await self._make_wc_request("products", params)
                
                if not data:
                    break
                
                for product_data in data:
                    product = WooCommerceProductData(
                        product_id=str(product_data["id"]),
                        name=product_data["name"],
                        slug=product_data["slug"],
                        permalink=product_data["permalink"],
                        date_created=datetime.fromisoformat(product_data["date_created"].replace("Z", "+00:00")),
                        date_modified=datetime.fromisoformat(product_data["date_modified"].replace("Z", "+00:00")),
                        type=product_data["type"],
                        status=product_data["status"],
                        featured=product_data["featured"],
                        catalog_visibility=product_data["catalog_visibility"],
                        description=product_data["description"],
                        short_description=product_data["short_description"],
                        sku=product_data.get("sku"),
                        price=product_data["price"],
                        regular_price=product_data["regular_price"],
                        sale_price=product_data["sale_price"],
                        date_on_sale_from=datetime.fromisoformat(product_data["date_on_sale_from"].replace("Z", "+00:00")) if product_data.get("date_on_sale_from") else None,
                        date_on_sale_to=datetime.fromisoformat(product_data["date_on_sale_to"].replace("Z", "+00:00")) if product_data.get("date_on_sale_to") else None,
                        price_html=product_data["price_html"],
                        on_sale=product_data["on_sale"],
                        purchasable=product_data["purchasable"],
                        total_sales=product_data["total_sales"],
                        virtual=product_data["virtual"],
                        downloadable=product_data["downloadable"],
                        downloads=product_data.get("downloads", []),
                        download_limit=product_data["download_limit"],
                        download_expiry=product_data["download_expiry"],
                        external_url=product_data["external_url"],
                        button_text=product_data["button_text"],
                        tax_status=product_data["tax_status"],
                        tax_class=product_data["tax_class"],
                        manage_stock=product_data["manage_stock"],
                        stock_quantity=product_data.get("stock_quantity"),
                        stock_status=product_data["stock_status"],
                        backorders=product_data["backorders"],
                        backorders_allowed=product_data["backorders_allowed"],
                        backordered=product_data["backordered"],
                        weight=product_data["weight"],
                        dimensions=product_data.get("dimensions", {}),
                        shipping_required=product_data["shipping_required"],
                        shipping_taxable=product_data["shipping_taxable"],
                        shipping_class=product_data["shipping_class"],
                        shipping_class_id=product_data["shipping_class_id"],
                        reviews_allowed=product_data["reviews_allowed"],
                        average_rating=product_data["average_rating"],
                        rating_count=product_data["rating_count"],
                        related_ids=product_data.get("related_ids", []),
                        upsell_ids=product_data.get("upsell_ids", []),
                        cross_sell_ids=product_data.get("cross_sell_ids", []),
                        parent_id=product_data["parent_id"],
                        purchase_note=product_data["purchase_note"],
                        categories=product_data.get("categories", []),
                        tags=product_data.get("tags", []),
                        images=product_data.get("images", []),
                        attributes=product_data.get("attributes", []),
                        default_attributes=product_data.get("default_attributes", []),
                        variations=product_data.get("variations", []),
                        grouped_products=product_data.get("grouped_products", []),
                        menu_order=product_data["menu_order"],
                        meta_data=product_data.get("meta_data", [])
                    )
                    products.append(product)
                
                if len(data) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Retrieved {len(products)} products from WooCommerce")
            return products[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get WooCommerce products: {str(e)}")
            raise
    
    async def get_wordpress_posts(self, date_start: date, date_end: date, limit: int = 100) -> List[WordPressPostData]:
        """Get WordPress posts for content attribution analysis"""
        try:
            posts = []
            page = 1
            per_page = min(limit, 100)
            
            while len(posts) < limit:
                params = {
                    "after": date_start.isoformat(),
                    "before": date_end.isoformat(),
                    "per_page": per_page,
                    "page": page,
                    "status": "publish"
                }
                
                data = await self._make_wp_request("posts", params)
                
                if not data:
                    break
                
                for post_data in data:
                    # Get additional analytics data if available
                    analytics_data = await self._get_post_analytics(post_data["id"])
                    
                    post = WordPressPostData(
                        post_id=str(post_data["id"]),
                        title=post_data["title"]["rendered"],
                        content=post_data["content"]["rendered"],
                        excerpt=post_data["excerpt"]["rendered"],
                        status=post_data["status"],
                        type=post_data["type"],
                        slug=post_data["slug"],
                        permalink=post_data["link"],
                        date=datetime.fromisoformat(post_data["date"].replace("Z", "+00:00")),
                        date_gmt=datetime.fromisoformat(post_data["date_gmt"].replace("Z", "+00:00")),
                        modified=datetime.fromisoformat(post_data["modified"].replace("Z", "+00:00")),
                        modified_gmt=datetime.fromisoformat(post_data["modified_gmt"].replace("Z", "+00:00")),
                        author=post_data["author"],
                        featured_media=post_data["featured_media"],
                        comment_status=post_data["comment_status"],
                        ping_status=post_data["ping_status"],
                        sticky=post_data["sticky"],
                        template=post_data["template"],
                        format=post_data["format"],
                        categories=post_data.get("categories", []),
                        tags=post_data.get("tags", []),
                        meta=post_data.get("meta", {}),
                        page_views=analytics_data.get("page_views", 0),
                        social_shares=analytics_data.get("social_shares", {}),
                        conversion_rate=analytics_data.get("conversion_rate", 0.0)
                    )
                    posts.append(post)
                
                if len(data) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Retrieved {len(posts)} WordPress posts")
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get WordPress posts: {str(e)}")
            return []  # Non-critical, return empty list
    
    async def _extract_attribution_data(self, order_data: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Extract UTM parameters and content attribution from order"""
        utm_params = {}
        content_attribution = {}
        
        # Check meta data for UTM parameters
        meta_data = order_data.get("meta_data", [])
        for meta in meta_data:
            key = meta.get("key", "")
            value = meta.get("value", "")
            
            if key.startswith("utm_"):
                utm_params[key] = value
            elif key in ["referring_post", "landing_page", "content_source"]:
                content_attribution[key] = value
        
        # Check customer note for UTM parameters
        customer_note = order_data.get("customer_note", "")
        if customer_note and "utm_" in customer_note:
            # Simple extraction from note
            for param in ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"]:
                if param in customer_note:
                    try:
                        start = customer_note.find(param + "=") + len(param) + 1
                        end = customer_note.find("&", start)
                        if end == -1:
                            end = customer_note.find(" ", start)
                        if end == -1:
                            end = len(customer_note)
                        utm_params[param] = customer_note[start:end].strip()
                    except Exception:
                        pass
        
        return utm_params, content_attribution
    
    async def _get_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Get analytics data for a WordPress post"""
        # This would integrate with analytics plugins like Google Analytics, Jetpack, etc.
        # For now, return mock data
        return {
            "page_views": 0,
            "social_shares": {},
            "conversion_rate": 0.0
        }
    
    async def transform_to_causal_format(
        self, 
        orders: List[WooCommerceOrderData], 
        org_id: str,
        posts: Optional[List[WordPressPostData]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[CausalMarketingData]:
        """Transform WooCommerce data to causal marketing format with content attribution"""
        causal_data_list = []
        
        # Create content attribution mapping
        content_map = {}
        if posts:
            for post in posts:
                content_map[post.post_id] = post
        
        for order in orders:
            try:
                # Create base causal marketing data
                causal_data = await self._create_causal_marketing_data(order, org_id, content_map, historical_data)
                
                # Enhance with KSE substrate integration
                await self._enhance_with_kse_substrate(causal_data, order, org_id, content_map)
                
                causal_data_list.append(causal_data)
                
            except Exception as e:
                logger.error(f"Failed to transform order {order.order_id} to causal format: {str(e)}")
                continue
        
        logger.info(f"Transformed {len(causal_data_list)} WooCommerce orders to causal format")
        return causal_data_list
    
    async def _create_causal_marketing_data(
        self, 
        order: WooCommerceOrderData, 
        org_id: str,
        content_map: Dict[str, WordPressPostData],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> CausalMarketingData:
        """Create causal marketing data from WooCommerce order"""
        
        # Determine treatment assignment
        treatment_assignment = await self._determine_treatment_assignment(order, content_map, historical_data)
        
        # Create calendar features
        calendar_features = await self.causal_transformer._create_calendar_features(order.date_created)
        
        # Detect confounders
        confounders = await self._detect_woocommerce_confounders(order, content_map, historical_data)
        
        # Extract geographic data
        geographic_data = self._extract_geographic_data(order)
        
        # Extract audience data
        audience_data = self._extract_audience_data(order)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(order)
        
        # Determine campaign info from content attribution
        campaign_id, campaign_name = self._determine_campaign_from_content(order, content_map)
        
        return CausalMarketingData(
            id=f"woocommerce_order_{order.order_id}_{uuid.uuid4().hex[:8]}",
            org_id=org_id,
            timestamp=order.date_created,
            data_source=DataSource.WOOCOMMERCE,
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            campaign_objective=self._map_campaign_objective(order.utm_parameters.get("utm_medium")),
            campaign_status=AdStatus.ACTIVE,
            treatment_group=treatment_assignment.treatment_group,
            treatment_type=treatment_assignment.treatment_type,
            treatment_intensity=treatment_assignment.treatment_intensity,
            randomization_unit=treatment_assignment.randomization_unit,
            experiment_id=treatment_assignment.experiment_id,
            spend=0.0,  # WooCommerce doesn't track ad spend directly
            impressions=0,  # Not available in WooCommerce
            clicks=1,  # Each order represents engagement
            conversions=1,  # Each order is a conversion
            revenue=order.total,
            platform_metrics={
                "order_number": order.order_number,
                "order_key": order.order_key,
                "status": order.status,
                "payment_method": order.payment_method,
                "payment_method_title": order.payment_method_title,
                "discount_total": order.discount_total,
                "shipping_total": order.shipping_total,
                "cart_tax": order.cart_tax,
                "total_tax": order.total_tax,
                "line_items_count": len(order.line_items),
                "coupon_lines": order.coupon_lines,
                "customer_ip_address": order.customer_ip_address,
                "customer_user_agent": order.customer_user_agent,
                "content_attribution": order.content_attribution
            },
            confounders=confounders,
            calendar_features=calendar_features,
            geographic_data=geographic_data,
            audience_data=audience_data,
            causal_metadata={
                "platform": "woocommerce",
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "utm_parameters": order.utm_parameters,
                "content_attribution": order.content_attribution,
                "wordpress_integration": True,
                "data_source_confidence": 0.9
            },
            data_quality_score=data_quality_score
        )
    
    async def _enhance_with_kse_substrate(
        self, 
        causal_data: CausalMarketingData, 
        order: WooCommerceOrderData, 
        org_id: str,
        content_map: Dict[str, WordPressPostData]
    ):
        """Enhance causal data with KSE universal substrate integration"""
        try:
            # 1. Neural Embeddings - Create rich content representation
            neural_content = self._create_neural_content(order, content_map)
            
            # 2. Conceptual Spaces - Map to conceptual dimensions
            conceptual_space = self._map_to_conceptual_space(order, content_map)
            
            # 3. Knowledge Graph - Create relationships including content
            knowledge_graph_nodes = self._create_knowledge_graph_nodes(order, causal_data, content_map)
            
            # Store in KSE with causal relationships including content attribution
            causal_relationships = [
                CausalRelationship(
                    source_node=f"customer_{order.customer_id}" if order.customer_id else f"guest_{order.billing.get('email', 'unknown')}",
                    target_node=f"order_{order.order_id}",
                    relationship_type="purchases",
                    strength=0.95,
                    confidence=0.9,
                    temporal_lag=0,
                    metadata={
                        "order_value": order.total,
                        "currency": order.currency,
                        "payment_method": order.payment_method
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
                        "attribution_model": "content_assisted",
                        "revenue": order.total
                    }
                )
            ]
            
            # Add content attribution relationships
            if order.content_attribution.get("referring_post"):
                causal_relationships.append(
                    CausalRelationship(
                        source_node=f"content_{order.content_attribution['referring_post']}",
                        target_node=f"order_{order.order_id}",
                        relationship_type="influences_purchase",
                        strength=0.7,
                        confidence=0.8,
                        temporal_lag=0,
                        metadata={
                            "content_type": "blog_post",
                            "attribution_type": "content_assisted"
                        }
                    )
                )
            
            # Create causal memory entry
            causal_memory = CausalMemoryEntry(
                content=neural_content,
                causal_relationships=causal_relationships,
                temporal_context={
                    "timestamp": order.date_created.isoformat(),
                    "day_of_week": order.date_created.weekday(),
                    "hour_of_day": order.date_created.hour,
                    "is_weekend": order.date_created.weekday() >= 5
                },
                causal_metadata={
                    "platform": "woocommerce",
                    "data_type": "e_commerce_order",
                    "treatment_type": causal_data.treatment_type.value,
                    "revenue_impact": order.total,
                    "content_attribution": bool(order.content_attribution)
                },
                platform_context={
                    "source": "woocommerce",
                    "order_id": order.order_id,
                    "customer_segment": self._determine_customer_segment(order),
                    "product_categories": self._extract_product_categories(order),
                    "content_context": self._extract_content_context(order, content_map)
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
            causal_data.causal_metadata["content_attribution_integrated"] = True
            
            logger.info(f"Enhanced WooCommerce order {order.order_id} with KSE substrate integration")
            
        except Exception as e:
            logger.error(f"Failed to enhance order {order.order_id} with KSE substrate: {str(e)}")
            # Continue without KSE enhancement
            causal_data.causal_metadata["kse_integration_failed"] = True
            causal_data.causal_metadata["kse_error"] = str(e)
    
    def _create_neural_content(self, order: WooCommerceOrderData, content_map: Dict[str, WordPressPostData]) -> str:
        """Create rich text content for neural embedding including WordPress content"""
        content_parts = [
            f"WooCommerce order {order.order_number} for ${order.total} {order.currency}",
            f"Customer: {order.billing.get('email', 'guest')}",
            f"Payment: {order.payment_method_title or order.payment_method}",
            f"Status: {order.status}",
            f"Items: {len(order.line_items)} products"
        ]
        
        # Add UTM information
        if order.utm_parameters:
            utm_info = ", ".join([f"{k}={v}" for k, v in order.utm_parameters.items() if v])
            if utm_info:
                content_parts.append(f"UTM: {utm_info}")
        
        # Add product information
        if order.line_items:
            product_names = [item.get("name", "Unknown") for item in order.line_items[:3]]
            content_parts.append(f"Products: {', '.join(product_names)}")
        
        # Add content attribution
        if order.content_attribution:
            referring_post = order.content_attribution.get("referring_post")
            if referring_post and referring_post in content_map:
                post = content_map[referring_post]
                content_parts.append(f"Influenced by content: {post.title}")
                content_parts.append(f"Content type: {post.type}")
        
        # Add customer information
        if order.customer_id:
            content_parts.append(f"Returning customer: {order.customer_id}")
        
        # Add discount information
        if order.discount_total > 0:
            content_parts.append(f"Discount applied: ${order.discount_total}")
        
        return ". ".join(content_parts)
    
    def _map_to_conceptual_space(self, order: WooCommerceOrderData, content_map: Dict[str, WordPressPostData]) -> str:
        """Map order to conceptual space dimensions including content influence"""
        dimensions = []
        
        # Price dimension
        if order.total < 50:
            dimensions.append("low_value")
        elif order.total < 200:
            dimensions.append("medium_value")
        else:
            dimensions.append("high_value")
        
        # Customer dimension
        if order.customer_id:
            dimensions.append("returning_customer")
        else:
            dimensions.append("new_customer")
        
        # Content influence dimension
        if order.content_attribution:
            dimensions.append("content_influenced")
        else:
            dimensions.append("direct_purchase")
        
        # Source dimension
        utm_medium = order.utm_parameters.get("utm_medium", "").lower()
        if utm_medium in ["cpc", "ppc", "paid"]:
            dimensions.append("paid_acquisition")
        elif utm_medium == "email":
            dimensions.append("email_marketing")
        elif utm_medium in ["social", "facebook", "instagram"]:
            dimensions.append("social_commerce")
        elif order.content_attribution:
            dimensions.append("content_marketing")
        else:
            dimensions.append("organic_commerce")
        
        return "_".join(dimensions)
    
    def _create_knowledge_graph_nodes(
        self,
        order: WooCommerceOrderData,
        causal_data: CausalMarketingData,
        content_map: Dict[str, WordPressPostData]
    ) -> List[str]:
        """Create knowledge graph node connections including content nodes"""
        nodes = [
            f"order_{order.order_id}",
            f"customer_{order.customer_id}" if order.customer_id else f"guest_{order.billing.get('email', 'unknown')}",
            f"campaign_{causal_data.campaign_id}"
        ]
        
        # Add product nodes
        for item in order.line_items[:5]:
            if item.get("product_id"):
                nodes.append(f"product_{item['product_id']}")
        
        # Add content nodes
        if order.content_attribution.get("referring_post"):
            nodes.append(f"content_{order.content_attribution['referring_post']}")
        
        # Add UTM nodes
        if order.utm_parameters.get("utm_source"):
            nodes.append(f"utm_source_{order.utm_parameters['utm_source']}")
        if order.utm_parameters.get("utm_medium"):
            nodes.append(f"utm_medium_{order.utm_parameters['utm_medium']}")
        
        # Add payment method node
        if order.payment_method:
            nodes.append(f"payment_{order.payment_method}")
        
        return nodes
    
    async def _determine_treatment_assignment(
        self,
        order: WooCommerceOrderData,
        content_map: Dict[str, WordPressPostData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for WooCommerce order"""
        
        # Analyze UTM parameters and content attribution
        utm_source = order.utm_parameters.get("utm_source", "")
        utm_medium = order.utm_parameters.get("utm_medium", "")
        utm_campaign = order.utm_parameters.get("utm_campaign", "")
        has_content_attribution = bool(order.content_attribution)
        
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
        elif has_content_attribution:
            treatment_type = TreatmentType.CREATIVE_CHANGE
            treatment_group = "content_marketing"
            treatment_intensity = 0.65
        elif utm_source:
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
        
        # Generate experiment ID
        experiment_id = None
        if utm_campaign:
            experiment_id = f"woocommerce_campaign_{utm_campaign}_{order.date_created.strftime('%Y%m')}"
        elif has_content_attribution:
            experiment_id = f"woocommerce_content_{order.date_created.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.85
        )
    
    async def _detect_woocommerce_confounders(
        self,
        order: WooCommerceOrderData,
        content_map: Dict[str, WordPressPostData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect WooCommerce-specific confounding variables"""
        confounders = []
        
        # Discount confounder
        if order.discount_total > 0:
            discount_rate = order.discount_total / (order.total + order.discount_total)
            confounders.append(ConfounderVariable(
                variable_name="discount_applied",
                variable_type="continuous",
                value=discount_rate,
                importance_score=0.8,
                detection_method="discount_analysis",
                control_strategy="include_discount_rate_as_covariate"
            ))
        
        # Payment method confounder
        if order.payment_method:
            confounders.append(ConfounderVariable(
                variable_name="payment_method",
                variable_type="categorical",
                value=order.payment_method,
                importance_score=0.6,
                detection_method="payment_analysis",
                control_strategy="include_payment_method_dummies"
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
        
        # Content influence confounder
        if order.content_attribution:
            confounders.append(ConfounderVariable(
                variable_name="content_influenced",
                variable_type="binary",
                value=True,
                importance_score=0.75,
                detection_method="content_attribution_analysis",
                control_strategy="include_content_influence_indicator"
            ))
        
        # Shipping cost confounder
        if order.shipping_total > 0:
            shipping_rate = order.shipping_total / order.total
            if shipping_rate > 0.1:  # More than 10% of order value
                confounders.append(ConfounderVariable(
                    variable_name="high_shipping_cost",
                    variable_type="continuous",
                    value=shipping_rate,
                    importance_score=0.65,
                    detection_method="shipping_analysis",
                    control_strategy="include_shipping_rate_as_covariate"
                ))
        
        return confounders
    
    def _extract_geographic_data(self, order: WooCommerceOrderData) -> Dict[str, Any]:
        """Extract geographic information from order"""
        geographic_data = {}
        
        # Billing address
        if order.billing:
            geographic_data.update({
                "billing_country": order.billing.get("country"),
                "billing_state": order.billing.get("state"),
                "billing_city": order.billing.get("city"),
                "billing_postcode": order.billing.get("postcode")
            })
        
        # Shipping address
        if order.shipping:
            geographic_data.update({
                "shipping_country": order.shipping.get("country"),
                "shipping_state": order.shipping.get("state"),
                "shipping_city": order.shipping.get("city"),
                "shipping_postcode": order.shipping.get("postcode")
            })
        
        return geographic_data
    
    def _extract_audience_data(self, order: WooCommerceOrderData) -> Dict[str, Any]:
        """Extract audience characteristics from order"""
        audience_data = {}
        
        # Customer data
        audience_data.update({
            "customer_type": "returning" if order.customer_id else "guest",
            "has_account": bool(order.customer_id),
            "order_value_segment": self._categorize_order_value(order.total),
            "cart_size": len(order.line_items),
            "used_coupon": len(order.coupon_lines) > 0,
            "discount_amount": order.discount_total,
            "shipping_required": order.shipping_total > 0,
            "payment_method": order.payment_method,
            "product_categories": self._extract_product_categories(order)
        })
        
        # Content attribution
        if order.content_attribution:
            audience_data.update({
                "content_influenced": True,
                "content_attribution": order.content_attribution
            })
        else:
            audience_data["content_influenced"] = False
        
        return audience_data
    
    def _calculate_data_quality_score(self, order: WooCommerceOrderData) -> float:
        """Calculate data quality score for causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [order.order_id, order.total, order.date_created, order.currency]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.25)
        
        # Customer data quality
        customer_score = 0.3  # Base score
        if order.customer_id:
            customer_score += 0.4
        if order.billing.get("email"):
            customer_score += 0.3
        score_components.append(min(customer_score, 1.0) * 0.2)
        
        # Attribution data quality
        attribution_score = 0.2  # Base score
        if order.utm_parameters:
            attribution_score += 0.4
        if order.content_attribution:
            attribution_score += 0.3
        if order.customer_user_agent:
            attribution_score += 0.1
        score_components.append(min(attribution_score, 1.0) * 0.3)
        
        # Product data quality
        product_score = 0.5 if order.line_items else 0.0
        if order.line_items:
            product_score += 0.3 if len(order.line_items) > 0 else 0.0
            product_score += 0.2 if any(item.get("product_id") for item in order.line_items) else 0.0
        score_components.append(min(product_score, 1.0) * 0.15)
        
        # Payment data quality
        payment_score = 0.5 if order.payment_method else 0.0
        if order.transaction_id:
            payment_score += 0.5
        score_components.append(min(payment_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _determine_campaign_from_content(self, order: WooCommerceOrderData, content_map: Dict[str, WordPressPostData]) -> tuple[str, str]:
        """Determine campaign ID and name from content attribution"""
        # Check UTM campaign first
        utm_campaign = order.utm_parameters.get("utm_campaign")
        if utm_campaign:
            return utm_campaign, f"Campaign: {utm_campaign}"
        
        # Check content attribution
        referring_post = order.content_attribution.get("referring_post")
        if referring_post and referring_post in content_map:
            post = content_map[referring_post]
            return f"content_{referring_post}", f"Content: {post.title}"
        
        # Check UTM source/medium
        utm_source = order.utm_parameters.get("utm_source")
        utm_medium = order.utm_parameters.get("utm_medium")
        if utm_source or utm_medium:
            campaign_parts = [part for part in [utm_source, utm_medium] if part]
            campaign_id = "_".join(campaign_parts)
            campaign_name = f"Traffic: {' / '.join(campaign_parts)}"
            return campaign_id, campaign_name
        
        # Default to organic
        return "woocommerce_organic", "WooCommerce Organic Traffic"
    
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
        elif utm_medium in ["content", "blog", "article"]:
            return CampaignObjective.ENGAGEMENT
        else:
            return CampaignObjective.CONVERSIONS
    
    def _determine_customer_segment(self, order: WooCommerceOrderData) -> str:
        """Determine customer segment based on order data"""
        if not order.customer_id:
            return "guest_customer"
        
        # This would ideally check historical order data
        if order.total > 300:
            return "high_value_customer"
        elif order.total > 100:
            return "medium_value_customer"
        else:
            return "regular_customer"
    
    def _extract_product_categories(self, order: WooCommerceOrderData) -> List[str]:
        """Extract product categories from line items"""
        categories = set()
        
        for item in order.line_items:
            # WooCommerce line items don't include category info directly
            # This would need to be enriched with product data
            product_name = item.get("name", "").lower()
            
            # Basic categorization based on product names
            if any(keyword in product_name for keyword in ["shirt", "dress", "pants", "clothing"]):
                categories.add("clothing")
            elif any(keyword in product_name for keyword in ["book", "ebook", "guide"]):
                categories.add("books")
            elif any(keyword in product_name for keyword in ["electronics", "phone", "computer", "tech"]):
                categories.add("electronics")
            elif any(keyword in product_name for keyword in ["home", "furniture", "decor"]):
                categories.add("home_goods")
            elif any(keyword in product_name for keyword in ["beauty", "cosmetic", "skincare"]):
                categories.add("beauty")
            else:
                categories.add("general")
        
        return list(categories)
    
    def _categorize_order_value(self, total: float) -> str:
        """Categorize order value into segments"""
        if total < 30:
            return "low_value"
        elif total < 100:
            return "medium_value"
        elif total < 300:
            return "high_value"
        else:
            return "premium_value"
    
    def _extract_content_context(self, order: WooCommerceOrderData, content_map: Dict[str, WordPressPostData]) -> Dict[str, Any]:
        """Extract content context for platform context"""
        content_context = {}
        
        referring_post = order.content_attribution.get("referring_post")
        if referring_post and referring_post in content_map:
            post = content_map[referring_post]
            content_context = {
                "post_id": post.post_id,
                "post_title": post.title,
                "post_type": post.type,
                "post_date": post.date.isoformat(),
                "post_categories": post.categories,
                "post_tags": post.tags,
                "page_views": post.page_views,
                "conversion_rate": post.conversion_rate
            }
        
        return content_context
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()