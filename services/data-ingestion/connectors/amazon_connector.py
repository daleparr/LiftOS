"""
Amazon Seller Central Connector with Marketplace Intelligence and KSE Universal Substrate
Tier 1 Marketplace Platform Connector for LiftOS v1.3.0
"""
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
import httpx
import json
from pydantic import BaseModel, Field
import uuid
import hmac
import hashlib
import urllib.parse

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


class AmazonSalesData(BaseModel):
    """Amazon sales data model"""
    order_id: str
    purchase_date: datetime
    last_updated_date: datetime
    order_status: str
    fulfillment_channel: str  # AFN (Amazon) or MFN (Merchant)
    sales_channel: str
    order_channel: str
    url: Optional[str] = None
    ship_service_level: str
    product_name: str
    sku: str
    asin: str
    item_status: str
    quantity: int
    currency: str
    item_price: float
    item_tax: float
    shipping_price: float
    shipping_tax: float
    gift_wrap_price: float
    gift_wrap_tax: float
    item_promotion_discount: float
    ship_promotion_discount: float
    ship_city: Optional[str] = None
    ship_state: Optional[str] = None
    ship_postal_code: Optional[str] = None
    ship_country: Optional[str] = None
    promotion_ids: List[str] = Field(default_factory=list)
    is_business_order: bool = False
    purchase_order_number: Optional[str] = None
    price_designation: Optional[str] = None
    
    # Marketplace Intelligence
    marketplace_id: str
    marketplace_name: str
    competitor_rank: Optional[int] = None
    category_rank: Optional[int] = None
    buy_box_percentage: Optional[float] = None
    session_percentage: Optional[float] = None


class AmazonAdvertisingData(BaseModel):
    """Amazon advertising data model"""
    campaign_id: str
    campaign_name: str
    campaign_type: str  # sponsoredProducts, sponsoredBrands, sponsoredDisplay
    campaign_status: str
    ad_group_id: str
    ad_group_name: str
    targeting_type: str  # manual, auto
    match_type: Optional[str] = None  # exact, phrase, broad
    keyword: Optional[str] = None
    asin: Optional[str] = None
    sku: Optional[str] = None
    date: date
    impressions: int
    clicks: int
    cost: float
    attributed_conversions_1d: int
    attributed_conversions_7d: int
    attributed_conversions_14d: int
    attributed_conversions_30d: int
    attributed_sales_1d: float
    attributed_sales_7d: float
    attributed_sales_14d: float
    attributed_sales_30d: float
    attributed_units_ordered_1d: int
    attributed_units_ordered_7d: int
    attributed_units_ordered_14d: int
    attributed_units_ordered_30d: int
    
    # Advanced Metrics
    acos: float = 0.0  # Advertising Cost of Sales
    roas: float = 0.0  # Return on Ad Spend
    cpc: float = 0.0   # Cost Per Click
    ctr: float = 0.0   # Click Through Rate
    cvr: float = 0.0   # Conversion Rate


class AmazonProductData(BaseModel):
    """Amazon product data model"""
    asin: str
    sku: str
    product_name: str
    brand: str
    manufacturer: str
    part_number: Optional[str] = None
    model: Optional[str] = None
    product_category: str
    product_subcategory: Optional[str] = None
    item_weight: Optional[float] = None
    item_dimensions: Dict[str, float] = Field(default_factory=dict)
    package_dimensions: Dict[str, float] = Field(default_factory=dict)
    color: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    material: Optional[str] = None
    
    # Pricing and Inventory
    list_price: float
    your_price: float
    landed_price: float
    condition: str
    quantity: int
    fulfillment_channel: str
    
    # Performance Metrics
    sales_rank: Optional[int] = None
    category_rank: Optional[int] = None
    reviews_count: int = 0
    average_rating: float = 0.0
    buy_box_eligible: bool = False
    buy_box_winner: bool = False
    
    # Marketplace Intelligence
    competitor_count: int = 0
    lowest_competitor_price: Optional[float] = None
    highest_competitor_price: Optional[float] = None
    price_competitiveness: Optional[str] = None  # low, competitive, high


class AmazonConnector:
    """Amazon Seller Central and Advertising API connector with marketplace intelligence"""
    
    def __init__(
        self, 
        access_key: str, 
        secret_key: str, 
        role_arn: str,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        marketplace_id: str = "ATVPDKIKX0DER",  # US marketplace
        region: str = "us-east-1"
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.role_arn = role_arn
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.marketplace_id = marketplace_id
        self.region = region
        
        # API endpoints
        self.sp_api_base = f"https://sellingpartnerapi-na.amazon.com"
        self.advertising_api_base = f"https://advertising-api.amazon.com"
        
        self.client = httpx.AsyncClient(timeout=60.0)
        self.access_token = None
        self.token_expires_at = None
        
        # Rate limiting
        self.rate_limit_remaining = 100
        self.rate_limit_reset = time.time()
        
        # KSE Integration
        self.causal_transformer = CausalDataTransformer()
    
    async def _get_access_token(self) -> str:
        """Get access token for Amazon APIs"""
        if self.access_token and self.token_expires_at and datetime.utcnow() < self.token_expires_at:
            return self.access_token
        
        try:
            # Get LWA (Login with Amazon) token
            token_url = "https://api.amazon.com/auth/o2/token"
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            
            response = await self.client.post(token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
            
            return self.access_token
            
        except Exception as e:
            logger.error(f"Failed to get Amazon access token: {str(e)}")
            raise
    
    def _create_aws_signature(self, method: str, url: str, headers: Dict[str, str], payload: str = "") -> Dict[str, str]:
        """Create AWS Signature Version 4 for SP-API requests"""
        # This is a simplified version - production would use boto3 or aws-requests-auth
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        date_stamp = timestamp[:8]
        
        # Create canonical request
        canonical_uri = urllib.parse.urlparse(url).path
        canonical_querystring = urllib.parse.urlparse(url).query
        canonical_headers = '\n'.join([f"{k.lower()}:{v}" for k, v in sorted(headers.items())]) + '\n'
        signed_headers = ';'.join([k.lower() for k in sorted(headers.keys())])
        payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        
        canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        
        # Create string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f"{date_stamp}/{self.region}/execute-api/aws4_request"
        string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        # Calculate signature
        signing_key = self._get_signature_key(self.secret_key, date_stamp, self.region, 'execute-api')
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Create authorization header
        authorization_header = f"{algorithm} Credential={self.access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
        
        headers.update({
            'Authorization': authorization_header,
            'X-Amz-Date': timestamp
        })
        
        return headers
    
    def _get_signature_key(self, key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
        """Generate signing key for AWS Signature Version 4"""
        k_date = hmac.new(('AWS4' + key).encode('utf-8'), date_stamp.encode('utf-8'), hashlib.sha256).digest()
        k_region = hmac.new(k_date, region_name.encode('utf-8'), hashlib.sha256).digest()
        k_service = hmac.new(k_region, service_name.encode('utf-8'), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
        return k_signing
    
    async def _make_sp_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Amazon SP-API"""
        await self._handle_rate_limiting()
        
        access_token = await self._get_access_token()
        url = f"{self.sp_api_base}{endpoint}"
        
        headers = {
            'x-amz-access-token': access_token,
            'Content-Type': 'application/json',
            'User-Agent': 'LiftOS/1.3.0 (Language=Python)'
        }
        
        # Add AWS signature
        headers = self._create_aws_signature('GET', url, headers)
        
        try:
            response = await self.client.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(5)
                return await self._make_sp_api_request(endpoint, params)
            else:
                logger.error(f"Amazon SP-API error: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            logger.error(f"Failed to make Amazon SP-API request: {str(e)}")
            raise
    
    async def _make_advertising_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Amazon Advertising API"""
        access_token = await self._get_access_token()
        url = f"{self.advertising_api_base}{endpoint}"
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Amazon-Advertising-API-ClientId': self.client_id,
            'Amazon-Advertising-API-Scope': self.marketplace_id
        }
        
        try:
            response = await self.client.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to make Amazon Advertising API request: {str(e)}")
            raise
    
    async def _handle_rate_limiting(self):
        """Handle Amazon API rate limiting"""
        if self.rate_limit_remaining <= 5:
            wait_time = max(0, 2 - (time.time() - self.rate_limit_reset))
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            self.rate_limit_remaining = 100
            self.rate_limit_reset = time.time()
    
    async def get_sales_data(self, date_start: date, date_end: date) -> List[AmazonSalesData]:
        """Get sales data from Amazon SP-API"""
        try:
            sales_data = []
            
            # Get orders using SP-API Orders endpoint
            params = {
                'MarketplaceIds': self.marketplace_id,
                'CreatedAfter': date_start.isoformat(),
                'CreatedBefore': date_end.isoformat(),
                'MaxResultsPerPage': 100
            }
            
            next_token = None
            while True:
                if next_token:
                    params['NextToken'] = next_token
                
                data = await self._make_sp_api_request('/orders/v0/orders', params)
                orders = data.get('payload', {}).get('Orders', [])
                
                if not orders:
                    break
                
                # Get order items for each order
                for order in orders:
                    order_id = order['AmazonOrderId']
                    order_items = await self._get_order_items(order_id)
                    
                    for item in order_items:
                        # Get marketplace intelligence
                        marketplace_intel = await self._get_marketplace_intelligence(item.get('ASIN'))
                        
                        sales_record = AmazonSalesData(
                            order_id=order_id,
                            purchase_date=datetime.fromisoformat(order['PurchaseDate'].replace('Z', '+00:00')),
                            last_updated_date=datetime.fromisoformat(order['LastUpdateDate'].replace('Z', '+00:00')),
                            order_status=order['OrderStatus'],
                            fulfillment_channel=order.get('FulfillmentChannel', 'MFN'),
                            sales_channel=order.get('SalesChannel', 'Amazon.com'),
                            order_channel=order.get('OrderChannel', 'Online'),
                            ship_service_level=order.get('ShipServiceLevel', 'Standard'),
                            product_name=item.get('Title', ''),
                            sku=item.get('SellerSKU', ''),
                            asin=item.get('ASIN', ''),
                            item_status=item.get('ItemStatus', ''),
                            quantity=item.get('QuantityOrdered', 0),
                            currency=item.get('ItemPrice', {}).get('CurrencyCode', 'USD'),
                            item_price=float(item.get('ItemPrice', {}).get('Amount', 0)),
                            item_tax=float(item.get('ItemTax', {}).get('Amount', 0)),
                            shipping_price=float(item.get('ShippingPrice', {}).get('Amount', 0)),
                            shipping_tax=float(item.get('ShippingTax', {}).get('Amount', 0)),
                            gift_wrap_price=float(item.get('GiftWrapPrice', {}).get('Amount', 0)),
                            gift_wrap_tax=float(item.get('GiftWrapTax', {}).get('Amount', 0)),
                            item_promotion_discount=float(item.get('PromotionDiscount', {}).get('Amount', 0)),
                            ship_promotion_discount=float(item.get('ShippingDiscount', {}).get('Amount', 0)),
                            ship_city=order.get('ShippingAddress', {}).get('City'),
                            ship_state=order.get('ShippingAddress', {}).get('StateOrRegion'),
                            ship_postal_code=order.get('ShippingAddress', {}).get('PostalCode'),
                            ship_country=order.get('ShippingAddress', {}).get('CountryCode'),
                            is_business_order=order.get('IsBusinessOrder', False),
                            purchase_order_number=order.get('PurchaseOrderNumber'),
                            marketplace_id=self.marketplace_id,
                            marketplace_name=order.get('MarketplaceName', 'Amazon.com'),
                            **marketplace_intel
                        )
                        sales_data.append(sales_record)
                
                next_token = data.get('payload', {}).get('NextToken')
                if not next_token:
                    break
            
            logger.info(f"Retrieved {len(sales_data)} sales records from Amazon")
            return sales_data
            
        except Exception as e:
            logger.error(f"Failed to get Amazon sales data: {str(e)}")
            raise
    
    async def _get_order_items(self, order_id: str) -> List[Dict[str, Any]]:
        """Get order items for a specific order"""
        try:
            data = await self._make_sp_api_request(f'/orders/v0/orders/{order_id}/orderItems')
            return data.get('payload', {}).get('OrderItems', [])
        except Exception as e:
            logger.error(f"Failed to get order items for {order_id}: {str(e)}")
            return []
    
    async def _get_marketplace_intelligence(self, asin: str) -> Dict[str, Any]:
        """Get marketplace intelligence for an ASIN"""
        # This would integrate with additional Amazon APIs or third-party services
        # For now, return mock data
        return {
            "competitor_rank": None,
            "category_rank": None,
            "buy_box_percentage": None,
            "session_percentage": None
        }
    
    async def get_advertising_data(self, date_start: date, date_end: date) -> List[AmazonAdvertisingData]:
        """Get advertising data from Amazon Advertising API"""
        try:
            advertising_data = []
            
            # Get campaigns
            campaigns_data = await self._make_advertising_api_request('/v2/sp/campaigns')
            
            for campaign in campaigns_data:
                campaign_id = campaign['campaignId']
                
                # Get campaign performance data
                report_data = await self._get_advertising_report(
                    campaign_id, date_start, date_end, 'campaigns'
                )
                
                for record in report_data:
                    # Calculate derived metrics
                    cost = float(record.get('cost', 0))
                    clicks = int(record.get('clicks', 0))
                    impressions = int(record.get('impressions', 0))
                    sales_7d = float(record.get('attributedSales7d', 0))
                    conversions_7d = int(record.get('attributedConversions7d', 0))
                    
                    acos = (cost / sales_7d * 100) if sales_7d > 0 else 0
                    roas = (sales_7d / cost) if cost > 0 else 0
                    cpc = (cost / clicks) if clicks > 0 else 0
                    ctr = (clicks / impressions * 100) if impressions > 0 else 0
                    cvr = (conversions_7d / clicks * 100) if clicks > 0 else 0
                    
                    ad_record = AmazonAdvertisingData(
                        campaign_id=campaign_id,
                        campaign_name=campaign['name'],
                        campaign_type=campaign.get('campaignType', 'sponsoredProducts'),
                        campaign_status=campaign['state'],
                        ad_group_id=record.get('adGroupId', ''),
                        ad_group_name=record.get('adGroupName', ''),
                        targeting_type=record.get('targetingType', 'manual'),
                        match_type=record.get('matchType'),
                        keyword=record.get('keyword'),
                        asin=record.get('asin'),
                        sku=record.get('sku'),
                        date=datetime.strptime(record['date'], '%Y-%m-%d').date(),
                        impressions=impressions,
                        clicks=clicks,
                        cost=cost,
                        attributed_conversions_1d=int(record.get('attributedConversions1d', 0)),
                        attributed_conversions_7d=conversions_7d,
                        attributed_conversions_14d=int(record.get('attributedConversions14d', 0)),
                        attributed_conversions_30d=int(record.get('attributedConversions30d', 0)),
                        attributed_sales_1d=float(record.get('attributedSales1d', 0)),
                        attributed_sales_7d=sales_7d,
                        attributed_sales_14d=float(record.get('attributedSales14d', 0)),
                        attributed_sales_30d=float(record.get('attributedSales30d', 0)),
                        attributed_units_ordered_1d=int(record.get('attributedUnitsOrdered1d', 0)),
                        attributed_units_ordered_7d=int(record.get('attributedUnitsOrdered7d', 0)),
                        attributed_units_ordered_14d=int(record.get('attributedUnitsOrdered14d', 0)),
                        attributed_units_ordered_30d=int(record.get('attributedUnitsOrdered30d', 0)),
                        acos=acos,
                        roas=roas,
                        cpc=cpc,
                        ctr=ctr,
                        cvr=cvr
                    )
                    advertising_data.append(ad_record)
            
            logger.info(f"Retrieved {len(advertising_data)} advertising records from Amazon")
            return advertising_data
            
        except Exception as e:
            logger.error(f"Failed to get Amazon advertising data: {str(e)}")
            return []  # Non-critical, return empty list
    
    async def _get_advertising_report(self, campaign_id: str, date_start: date, date_end: date, report_type: str) -> List[Dict[str, Any]]:
        """Get advertising report data"""
        # This would use the Amazon Advertising API reporting endpoints
        # For now, return mock data
        return []
    
    async def transform_to_causal_format(
        self, 
        sales_data: List[AmazonSalesData], 
        advertising_data: List[AmazonAdvertisingData],
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[CausalMarketingData]:
        """Transform Amazon data to causal marketing format with marketplace intelligence"""
        causal_data_list = []
        
        # Create advertising lookup for attribution
        ad_lookup = {}
        for ad_record in advertising_data:
            key = f"{ad_record.asin}_{ad_record.date}"
            ad_lookup[key] = ad_record
        
        for sales_record in sales_data:
            try:
                # Find corresponding advertising data
                ad_data = ad_lookup.get(f"{sales_record.asin}_{sales_record.purchase_date.date()}")
                
                # Create causal marketing data
                causal_data = await self._create_causal_marketing_data(
                    sales_record, ad_data, org_id, historical_data
                )
                
                # Enhance with KSE substrate integration
                await self._enhance_with_kse_substrate(causal_data, sales_record, ad_data, org_id)
                
                causal_data_list.append(causal_data)
                
            except Exception as e:
                logger.error(f"Failed to transform sales record {sales_record.order_id} to causal format: {str(e)}")
                continue
        
        logger.info(f"Transformed {len(causal_data_list)} Amazon records to causal format")
        return causal_data_list
    
    async def _create_causal_marketing_data(
        self, 
        sales_record: AmazonSalesData, 
        ad_data: Optional[AmazonAdvertisingData],
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> CausalMarketingData:
        """Create causal marketing data from Amazon sales and advertising data"""
        
        # Determine treatment assignment
        treatment_assignment = await self._determine_treatment_assignment(sales_record, ad_data, historical_data)
        
        # Create calendar features
        calendar_features = await self.causal_transformer._create_calendar_features(sales_record.purchase_date)
        
        # Detect confounders
        confounders = await self._detect_amazon_confounders(sales_record, ad_data, historical_data)
        
        # Extract geographic data
        geographic_data = self._extract_geographic_data(sales_record)
        
        # Extract audience data
        audience_data = self._extract_audience_data(sales_record, ad_data)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(sales_record, ad_data)
        
        # Determine campaign info
        campaign_id, campaign_name = self._determine_campaign_info(sales_record, ad_data)
        
        return CausalMarketingData(
            id=f"amazon_sale_{sales_record.order_id}_{sales_record.sku}_{uuid.uuid4().hex[:8]}",
            org_id=org_id,
            timestamp=sales_record.purchase_date,
            data_source=DataSource.AMAZON_SELLER_CENTRAL,
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            campaign_objective=CampaignObjective.CONVERSIONS,
            campaign_status=AdStatus.ACTIVE if ad_data and ad_data.campaign_status == 'enabled' else AdStatus.PAUSED,
            treatment_group=treatment_assignment.treatment_group,
            treatment_type=treatment_assignment.treatment_type,
            treatment_intensity=treatment_assignment.treatment_intensity,
            randomization_unit=treatment_assignment.randomization_unit,
            experiment_id=treatment_assignment.experiment_id,
            spend=ad_data.cost if ad_data else 0.0,
            impressions=ad_data.impressions if ad_data else 0,
            clicks=ad_data.clicks if ad_data else 1,  # Sale implies at least one interaction
            conversions=1,  # Each sale is a conversion
            revenue=sales_record.item_price,
            platform_metrics={
                "order_id": sales_record.order_id,
                "asin": sales_record.asin,
                "sku": sales_record.sku,
                "fulfillment_channel": sales_record.fulfillment_channel,
                "marketplace_id": sales_record.marketplace_id,
                "quantity": sales_record.quantity,
                "item_tax": sales_record.item_tax,
                "shipping_price": sales_record.shipping_price,
                "promotion_discount": sales_record.item_promotion_discount,
                "is_business_order": sales_record.is_business_order,
                "buy_box_percentage": sales_record.buy_box_percentage,
                "competitor_rank": sales_record.competitor_rank,
                "category_rank": sales_record.category_rank,
                # Advertising metrics if available
                "acos": ad_data.acos if ad_data else None,
                "roas": ad_data.roas if ad_data else None,
                "cpc": ad_data.cpc if ad_data else None,
                "ctr": ad_data.ctr if ad_data else None,
                "cvr": ad_data.cvr if ad_data else None
            },
            confounders=confounders,
            calendar_features=calendar_features,
            geographic_data=geographic_data,
            audience_data=audience_data,
            causal_metadata={
                "platform": "amazon_seller_central",
                "order_id": sales_record.order_id,
                "asin": sales_record.asin,
                "marketplace_intelligence": True,
                "advertising_attribution": bool(ad_data),
                "fulfillment_channel": sales_record.fulfillment_channel,
                "data_source_confidence": 0.95
            },
            data_quality_score=data_quality_score
        )
    
    async def _enhance_with_kse_substrate(
        self, 
        causal_data: CausalMarketingData, 
        sales_record: AmazonSalesData, 
        ad_data: Optional[AmazonAdvertisingData],
        org_id: str
    ):
        """Enhance causal data with KSE universal substrate integration"""
        try:
            # 1. Neural Embeddings - Create rich marketplace content
            neural_content = self._create_neural_content(sales_record, ad_data)
            
            # 2. Conceptual Spaces - Map to marketplace dimensions
            conceptual_space = self._map_to_conceptual_space(sales_record, ad_data)
            
            # 3. Knowledge Graph - Create marketplace relationships
            knowledge_graph_nodes = self._create_knowledge_graph_nodes(sales_record, ad_data, causal_data)
            
            # Store in KSE with marketplace-specific causal relationships
            causal_relationships = [
                CausalRelationship(
                    source_node=f"product_{sales_record.asin}",
                    target_node=f"sale_{sales_record.order_id}",
                    relationship_type="generates_sale",
                    strength=0.95,
                    confidence=0.9,
                    temporal_lag=0,
                    metadata={
                        "sale_value": sales_record.item_price,
                        "quantity": sales_record.quantity,
                        "marketplace": sales_record.marketplace_name
                    }
                ),
                CausalRelationship(
                    source_node=f"marketplace_{sales_record.marketplace_id}",
                    target_node=f"sale_{sales_record.order_id}",
                    relationship_type="facilitates_sale",
                    strength=0.9,
                    confidence=0.85,
                    temporal_lag=0,
                    metadata={
                        "marketplace_name": sales_record.marketplace_name,
                        "fulfillment_channel": sales_record.fulfillment_channel
                    }
                )
            ]
            
            # Add advertising relationships if available
            if ad_data:
                causal_relationships.append(
                    CausalRelationship(
                        source_node=f"campaign_{ad_data.campaign_id}",
                        target_node=f"sale_{sales_record.order_id}",
                        relationship_type="drives_conversion",
                        strength=0.8,
                        confidence=0.9,
                        temporal_lag=0,
                        metadata={
                            "acos": ad_data.acos,
                            "roas": ad_data.roas,
                            "campaign_type": ad_data.campaign_type
                        }
                    )
                )
            
            # Create causal memory entry
            causal_memory = CausalMemoryEntry(
                content=neural_content,
                causal_relationships=causal_relationships,
                temporal_context={
                    "timestamp": sales_record.purchase_date.isoformat(),
                    "day_of_week": sales_record.purchase_date.weekday(),
                    "hour_of_day": sales_record.purchase_date.hour,
                    "is_weekend": sales_record.purchase_date.weekday() >= 5
                },
                causal_metadata={
                    "platform": "amazon_seller_central",
                    "data_type": "marketplace_sale",
                    "treatment_type": causal_data.treatment_type.value,
                    "revenue_impact": sales_record.item_price,
                    "marketplace_intelligence": True,
                    "advertising_attribution": bool(ad_data)
                },
                platform_context={
                    "source": "amazon",
                    "order_id": sales_record.order_id,
                    "asin": sales_record.asin,
                    "marketplace_id": sales_record.marketplace_id,
                    "fulfillment_channel": sales_record.fulfillment_channel,
                    "customer_segment": self._determine_customer_segment(sales_record),
                    "product_category": self._extract_product_category(sales_record),
                    "marketplace_intelligence": {
                        "buy_box_percentage": sales_record.buy_box_percentage,
                        "competitor_rank": sales_record.competitor_rank,
                        "category_rank": sales_record.category_rank
                    }
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
            causal_data.causal_metadata["marketplace_intelligence_integrated"] = True
            
            logger.info(f"Enhanced Amazon sale {sales_record.order_id} with KSE substrate integration")
            
        except Exception as e:
            logger.error(f"Failed to enhance sale {sales_record.order_id} with KSE substrate: {str(e)}")
            # Continue without KSE enhancement
            causal_data.causal_metadata["kse_integration_failed"] = True
            causal_data.causal_metadata["kse_error"] = str(e)
    
    def _create_neural_content(self, sales_record: AmazonSalesData, ad_data: Optional[AmazonAdvertisingData]) -> str:
        """Create rich text content for neural embedding including marketplace intelligence"""
        content_parts = [
            f"Amazon sale {sales_record.order_id} for {sales_record.product_name}",
            f"ASIN: {sales_record.asin}, SKU: {sales_record.sku}",
            f"Price: ${sales_record.item_price} {sales_record.currency}",
            f"Quantity: {sales_record.quantity}",
            f"Fulfillment: {sales_record.fulfillment_channel}",
            f"Marketplace: {sales_record.marketplace_name}"
        ]
        
        # Add marketplace intelligence
        if sales_record.buy_box_percentage:
            content_parts.append(f"Buy Box: {sales_record.buy_box_percentage:.1f}%")
        if sales_record.category_rank:
            content_parts.append(f"Category rank: #{sales_record.category_rank}")
        
        # Add advertising information
        if ad_data:
            content_parts.extend([
                f"Campaign: {ad_data.campaign_name}",
                f"Ad spend: ${ad_data.cost:.2f}",
                f"ACOS: {ad_data.acos:.1f}%",
                f"ROAS: {ad_data.roas:.2f}x"
            ])
        
        # Add geographic information
        if sales_record.ship_country:
            location_parts = [part for part in [
                sales_record.ship_city,
                sales_record.ship_state,
                sales_record.ship_country
            ] if part]
            if location_parts:
                content_parts.append(f"Shipped to: {', '.join(location_parts)}")
        
        # Add promotion information
        if sales_record.item_promotion_discount > 0:
            content_parts.append(f"Promotion discount: ${sales_record.item_promotion_discount}")
        
        return ". ".join(content_parts)
    
    def _map_to_conceptual_space(self, sales_record: AmazonSalesData, ad_data: Optional[AmazonAdvertisingData]) -> str:
        """Map sale to conceptual space dimensions including marketplace factors"""
        dimensions = []
        
        # Price dimension
        if sales_record.item_price < 25:
            dimensions.append("low_value")
        elif sales_record.item_price < 100:
            dimensions.append("medium_value")
        else:
            dimensions.append("high_value")
        
        # Fulfillment dimension
        if sales_record.fulfillment_channel == "AFN":
            dimensions.append("amazon_fulfilled")
        else:
            dimensions.append("merchant_fulfilled")
        
        # Advertising dimension
        if ad_data:
            if ad_data.acos < 20:
                dimensions.append("efficient_advertising")
            elif ad_data.acos < 50:
                dimensions.append("moderate_advertising")
            else:
                dimensions.append("expensive_advertising")
        else:
            dimensions.append("organic_sale")
        
        # Marketplace competitiveness
        if sales_record.buy_box_percentage and sales_record.buy_box_percentage > 80:
            dimensions.append("dominant_position")
        elif sales_record.buy_box_percentage and sales_record.buy_box_percentage > 50:
            dimensions.append("competitive_position")
        else:
            dimensions.append("challenging_position")
        
        # Customer type
        if sales_record.is_business_order:
            dimensions.append("b2b_customer")
        else:
            dimensions.append("b2c_customer")
        
        return "_".join(dimensions)
    
    def _create_knowledge_graph_nodes(
        self,
        sales_record: AmazonSalesData,
        ad_data: Optional[AmazonAdvertisingData],
        causal_data: CausalMarketingData
    ) -> List[str]:
        """Create knowledge graph node connections including marketplace entities"""
        nodes = [
            f"sale_{sales_record.order_id}",
            f"product_{sales_record.asin}",
            f"sku_{sales_record.sku}",
            f"marketplace_{sales_record.marketplace_id}",
            f"fulfillment_{sales_record.fulfillment_channel}"
        ]
        
        # Add advertising nodes
        if ad_data:
            nodes.extend([
                f"campaign_{ad_data.campaign_id}",
                f"ad_group_{ad_data.ad_group_id}",
                f"campaign_type_{ad_data.campaign_type}"
            ])
            
            if ad_data.keyword:
                nodes.append(f"keyword_{ad_data.keyword}")
        
        # Add geographic nodes
        if sales_record.ship_country:
            nodes.append(f"country_{sales_record.ship_country}")
        if sales_record.ship_state:
            nodes.append(f"state_{sales_record.ship_state}")
        
        # Add business type node
        if sales_record.is_business_order:
            nodes.append("customer_type_business")
        else:
            nodes.append("customer_type_consumer")
        
        return nodes
    
    async def _determine_treatment_assignment(
        self,
        sales_record: AmazonSalesData,
        ad_data: Optional[AmazonAdvertisingData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for Amazon sale"""
        
        # Determine treatment based on advertising data
        if ad_data:
            if ad_data.campaign_type == "sponsoredProducts":
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "sponsored_products"
                treatment_intensity = min(ad_data.acos / 100, 1.0)  # Normalize ACOS
            elif ad_data.campaign_type == "sponsoredBrands":
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "sponsored_brands"
                treatment_intensity = 0.8
            elif ad_data.campaign_type == "sponsoredDisplay":
                treatment_type = TreatmentType.TARGETING_CHANGE
                treatment_group = "sponsored_display"
                treatment_intensity = 0.7
            else:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "amazon_advertising"
                treatment_intensity = 0.6
        else:
            treatment_type = TreatmentType.CONTROL
            treatment_group = "organic_amazon"
            treatment_intensity = 0.0
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.PRODUCT_CATEGORY
        
        # Generate experiment ID
        experiment_id = None
        if ad_data:
            experiment_id = f"amazon_campaign_{ad_data.campaign_id}_{sales_record.purchase_date.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.9 if ad_data else 0.7
        )
    
    async def _detect_amazon_confounders(
        self,
        sales_record: AmazonSalesData,
        ad_data: Optional[AmazonAdvertisingData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect Amazon-specific confounding variables"""
        confounders = []
        
        # Buy Box percentage confounder
        if sales_record.buy_box_percentage is not None:
            confounders.append(ConfounderVariable(
                variable_name="buy_box_percentage",
                variable_type="continuous",
                value=sales_record.buy_box_percentage,
                importance_score=0.9,
                detection_method="marketplace_intelligence",
                control_strategy="include_buy_box_percentage_as_covariate"
            ))
        
        # Fulfillment channel confounder
        confounders.append(ConfounderVariable(
            variable_name="fulfillment_channel",
            variable_type="categorical",
            value=sales_record.fulfillment_channel,
            importance_score=0.8,
            detection_method="fulfillment_analysis",
            control_strategy="include_fulfillment_channel_dummies"
        ))
        
        # Promotion discount confounder
        if sales_record.item_promotion_discount > 0:
            discount_rate = sales_record.item_promotion_discount / sales_record.item_price
            confounders.append(ConfounderVariable(
                variable_name="promotion_discount",
                variable_type="continuous",
                value=discount_rate,
                importance_score=0.85,
                detection_method="promotion_analysis",
                control_strategy="include_discount_rate_as_covariate"
            ))
        
        # Business order confounder
        if sales_record.is_business_order:
            confounders.append(ConfounderVariable(
                variable_name="business_order",
                variable_type="binary",
                value=True,
                importance_score=0.7,
                detection_method="order_type_analysis",
                control_strategy="include_business_order_indicator"
            ))
        
        # Category rank confounder
        if sales_record.category_rank:
            confounders.append(ConfounderVariable(
                variable_name="category_rank",
                variable_type="continuous",
                value=sales_record.category_rank,
                importance_score=0.75,
                detection_method="marketplace_ranking",
                control_strategy="include_category_rank_as_covariate"
            ))
        
        return confounders
    
    def _extract_geographic_data(self, sales_record: AmazonSalesData) -> Dict[str, Any]:
        """Extract geographic information from sale"""
        return {
            "ship_city": sales_record.ship_city,
            "ship_state": sales_record.ship_state,
            "ship_postal_code": sales_record.ship_postal_code,
            "ship_country": sales_record.ship_country,
            "marketplace_id": sales_record.marketplace_id,
            "marketplace_name": sales_record.marketplace_name
        }
    
    def _extract_audience_data(self, sales_record: AmazonSalesData, ad_data: Optional[AmazonAdvertisingData]) -> Dict[str, Any]:
        """Extract audience characteristics from sale"""
        audience_data = {
            "customer_type": "business" if sales_record.is_business_order else "consumer",
            "fulfillment_preference": sales_record.fulfillment_channel,
            "order_value_segment": self._categorize_order_value(sales_record.item_price),
            "quantity_segment": self._categorize_quantity(sales_record.quantity),
            "has_promotion": sales_record.item_promotion_discount > 0,
            "marketplace": sales_record.marketplace_name,
            "product_category": self._extract_product_category(sales_record)
        }
        
        # Add advertising audience data
        if ad_data:
            audience_data.update({
                "advertising_influenced": True,
                "campaign_type": ad_data.campaign_type,
                "targeting_type": ad_data.targeting_type,
                "match_type": ad_data.match_type,
                "acos_segment": self._categorize_acos(ad_data.acos)
            })
        else:
            audience_data["advertising_influenced"] = False
        
        return audience_data
    
    def _calculate_data_quality_score(self, sales_record: AmazonSalesData, ad_data: Optional[AmazonAdvertisingData]) -> float:
        """Calculate data quality score for causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [sales_record.order_id, sales_record.asin, sales_record.item_price, sales_record.purchase_date]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Marketplace intelligence quality
        intel_score = 0.5  # Base score
        if sales_record.buy_box_percentage is not None:
            intel_score += 0.3
        if sales_record.category_rank is not None:
            intel_score += 0.2
        score_components.append(min(intel_score, 1.0) * 0.25)
        
        # Advertising attribution quality
        ad_score = 0.3 if ad_data else 0.1  # Lower score for organic sales
        if ad_data:
            ad_score += 0.4 if ad_data.cost > 0 else 0.0
            ad_score += 0.3 if ad_data.impressions > 0 else 0.0
        score_components.append(min(ad_score, 1.0) * 0.25)
        
        # Geographic data quality
        geo_score = 0.2  # Base score
        if sales_record.ship_country:
            geo_score += 0.4
        if sales_record.ship_state:
            geo_score += 0.2
        if sales_record.ship_city:
            geo_score += 0.2
        score_components.append(min(geo_score, 1.0) * 0.1)
        
        # Product data quality
        product_score = 0.5 if sales_record.sku else 0.2
        product_score += 0.3 if sales_record.product_name else 0.0
        product_score += 0.2 if sales_record.asin else 0.0
        score_components.append(min(product_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _determine_campaign_info(self, sales_record: AmazonSalesData, ad_data: Optional[AmazonAdvertisingData]) -> tuple[str, str]:
        """Determine campaign ID and name"""
        if ad_data:
            return ad_data.campaign_id, ad_data.campaign_name
        else:
            return f"amazon_organic_{sales_record.marketplace_id}", f"Amazon Organic - {sales_record.marketplace_name}"
    
    def _determine_customer_segment(self, sales_record: AmazonSalesData) -> str:
        """Determine customer segment based on sale data"""
        if sales_record.is_business_order:
            return "business_customer"
        elif sales_record.item_price > 200:
            return "high_value_consumer"
        elif sales_record.quantity > 3:
            return "bulk_buyer"
        else:
            return "regular_consumer"
    
    def _extract_product_category(self, sales_record: AmazonSalesData) -> str:
        """Extract product category from sale data"""
        # This would ideally use Amazon's product catalog API
        # For now, use basic categorization based on product name
        product_name = sales_record.product_name.lower()
        
        if any(keyword in product_name for keyword in ["book", "kindle", "ebook"]):
            return "books"
        elif any(keyword in product_name for keyword in ["electronics", "phone", "computer", "tech"]):
            return "electronics"
        elif any(keyword in product_name for keyword in ["clothing", "shirt", "dress", "shoes"]):
            return "clothing"
        elif any(keyword in product_name for keyword in ["home", "kitchen", "furniture"]):
            return "home_garden"
        elif any(keyword in product_name for keyword in ["beauty", "health", "personal"]):
            return "health_beauty"
        else:
            return "general"
    
    def _categorize_order_value(self, item_price: float) -> str:
        """Categorize order value into segments"""
        if item_price < 20:
            return "low_value"
        elif item_price < 75:
            return "medium_value"
        elif item_price < 200:
            return "high_value"
        else:
            return "premium_value"
    
    def _categorize_quantity(self, quantity: int) -> str:
        """Categorize quantity into segments"""
        if quantity == 1:
            return "single_item"
        elif quantity <= 3:
            return "small_quantity"
        elif quantity <= 10:
            return "medium_quantity"
        else:
            return "bulk_order"
    
    def _categorize_acos(self, acos: float) -> str:
        """Categorize ACOS into efficiency segments"""
        if acos < 15:
            return "highly_efficient"
        elif acos < 30:
            return "efficient"
        elif acos < 50:
            return "moderate"
        else:
            return "inefficient"
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()