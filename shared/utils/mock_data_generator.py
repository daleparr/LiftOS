"""
Mock Data Generator
Generates realistic mock marketing data for testing and fallback scenarios
"""

import random
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Import DataSource enum for the MockDataGenerator class
try:
    from shared.models.marketing import DataSource
except ImportError:
    # Fallback if import fails
    from enum import Enum
    class DataSource(str, Enum):
        META_BUSINESS = "meta_business"
        GOOGLE_ADS = "google_ads"
        KLAVIYO = "klaviyo"
        SHOPIFY = "shopify"
        WOOCOMMERCE = "woocommerce"
        AMAZON_SELLER_CENTRAL = "amazon_seller_central"
        HUBSPOT = "hubspot"
        SALESFORCE = "salesforce"
        STRIPE = "stripe"
        PAYPAL = "paypal"
        TIKTOK = "tiktok"
        SNOWFLAKE = "snowflake"
        DATABRICKS = "databricks"
        ZOHO_CRM = "zoho_crm"
        LINKEDIN_ADS = "linkedin_ads"
        X_ADS = "x_ads"

def generate_mock_marketing_data(
    platform: str,
    start_date: date,
    end_date: date,
    campaigns: Optional[int] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Generate mock marketing data for a platform"""
    
    if campaigns is None:
        campaigns = random.randint(3, 8)
    
    data = []
    date_range = (end_date - start_date).days + 1
    
    for i in range(campaigns):
        campaign_id = f"mock_{platform}_{i+1}_{uuid.uuid4().hex[:8]}"
        campaign_name = f"Mock Campaign {i+1} - {platform.replace('_', ' ').title()}"
        
        # Generate daily data for the date range
        for day_offset in range(date_range):
            current_date = start_date + timedelta(days=day_offset)
            
            # Platform-specific data generation
            if platform == 'meta_business':
                record = generate_meta_business_data(campaign_id, campaign_name, current_date)
            elif platform == 'google_ads':
                record = generate_google_ads_data(campaign_id, campaign_name, current_date)
            elif platform == 'klaviyo':
                record = generate_klaviyo_data(campaign_id, campaign_name, current_date)
            elif platform == 'shopify':
                record = generate_shopify_data(campaign_id, campaign_name, current_date)
            elif platform == 'woocommerce':
                record = generate_woocommerce_data(campaign_id, campaign_name, current_date)
            elif platform == 'amazon_seller_central':
                record = generate_amazon_data(campaign_id, campaign_name, current_date)
            elif platform == 'hubspot':
                record = generate_hubspot_data(campaign_id, campaign_name, current_date)
            elif platform == 'salesforce':
                record = generate_salesforce_data(campaign_id, campaign_name, current_date)
            elif platform == 'stripe':
                record = generate_stripe_data(campaign_id, campaign_name, current_date)
            elif platform == 'paypal':
                record = generate_paypal_data(campaign_id, campaign_name, current_date)
            elif platform == 'tiktok':
                record = generate_tiktok_data(campaign_id, campaign_name, current_date)
            elif platform == 'snowflake':
                record = generate_snowflake_data(campaign_id, campaign_name, current_date)
            elif platform == 'databricks':
                record = generate_databricks_data(campaign_id, campaign_name, current_date)
            elif platform == 'zoho_crm':
                record = generate_zoho_crm_data(campaign_id, campaign_name, current_date)
            elif platform == 'linkedin_ads':
                record = generate_linkedin_ads_data(campaign_id, campaign_name, current_date)
            elif platform == 'x_ads':
                record = generate_x_ads_data(campaign_id, campaign_name, current_date)
            else:
                record = generate_generic_data(campaign_id, campaign_name, current_date, platform)
            
            data.append(record)
    
    return data

def generate_meta_business_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Meta Business data"""
    base_spend = random.uniform(100, 1000)
    impressions = int(base_spend * random.uniform(800, 1200))
    clicks = int(impressions * random.uniform(0.01, 0.05))
    reach = int(impressions * random.uniform(0.6, 0.9))
    
    return {
        "id": f"meta_{campaign_id}_{date.isoformat()}",
        "account_id": f"act_{random.randint(100000000, 999999999)}",
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "status": random.choice(["ACTIVE", "PAUSED"]),
        "objective": random.choice(["CONVERSIONS", "TRAFFIC", "REACH", "BRAND_AWARENESS"]),
        "spend": round(base_spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "reach": reach,
        "frequency": round(impressions / reach, 2) if reach > 0 else 0,
        "cpm": round((base_spend / impressions) * 1000, 2) if impressions > 0 else 0,
        "cpc": round(base_spend / clicks, 2) if clicks > 0 else 0,
        "ctr": round((clicks / impressions) * 100, 2) if impressions > 0 else 0,
        "video_views": int(clicks * random.uniform(0.3, 0.8)),
        "actions": random.randint(0, int(clicks * 0.1)),
        "created_time": datetime.now().isoformat(),
        "updated_time": datetime.now().isoformat(),
        "mock_data": True
    }

def generate_google_ads_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Google Ads data"""
    base_cost = random.uniform(80, 800)
    impressions = int(base_cost * random.uniform(1000, 1500))
    clicks = int(impressions * random.uniform(0.02, 0.06))
    
    return {
        "id": f"google_{campaign_id}_{date.isoformat()}",
        "customer_id": f"{random.randint(1000000000, 9999999999)}",
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "status": random.choice(["ENABLED", "PAUSED", "REMOVED"]),
        "campaign_type": random.choice(["SEARCH", "DISPLAY", "SHOPPING", "VIDEO"]),
        "cost_micros": int(base_cost * 1000000),
        "cost": round(base_cost, 2),
        "impressions": impressions,
        "clicks": clicks,
        "ctr": round((clicks / impressions) * 100, 2) if impressions > 0 else 0,
        "average_cpc": round(base_cost / clicks, 2) if clicks > 0 else 0,
        "conversions": random.randint(0, int(clicks * 0.05)),
        "conversion_value": round(random.uniform(0, base_cost * 2), 2),
        "quality_score": round(random.uniform(6, 10), 1),
        "search_impression_share": round(random.uniform(0.4, 0.9), 2),
        "mock_data": True
    }

def generate_klaviyo_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Klaviyo email data"""
    recipients = random.randint(1000, 10000)
    delivered = int(recipients * random.uniform(0.95, 0.99))
    opened = int(delivered * random.uniform(0.15, 0.35))
    clicked = int(opened * random.uniform(0.1, 0.25))
    
    return {
        "id": f"klaviyo_{campaign_id}_{date.isoformat()}",
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "status": random.choice(["sent", "draft", "scheduled"]),
        "campaign_type": random.choice(["regular", "automation", "flow"]),
        "recipients": recipients,
        "delivered": delivered,
        "bounced": recipients - delivered,
        "opened": opened,
        "clicked": clicked,
        "unsubscribed": random.randint(0, int(recipients * 0.01)),
        "open_rate": round((opened / delivered) * 100, 2) if delivered > 0 else 0,
        "click_rate": round((clicked / delivered) * 100, 2) if delivered > 0 else 0,
        "revenue": round(random.uniform(0, clicked * 50), 2),
        "mock_data": True
    }

def generate_shopify_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Shopify e-commerce data"""
    orders = random.randint(10, 100)
    revenue = round(orders * random.uniform(50, 200), 2)
    
    return {
        "id": f"shopify_{campaign_id}_{date.isoformat()}",
        "shop_id": campaign_id,
        "shop_name": campaign_name,
        "date": date.isoformat(),
        "orders": orders,
        "revenue": revenue,
        "average_order_value": round(revenue / orders, 2) if orders > 0 else 0,
        "total_sales": revenue,
        "refunds": round(random.uniform(0, revenue * 0.05), 2),
        "shipping": round(random.uniform(0, revenue * 0.1), 2),
        "taxes": round(random.uniform(0, revenue * 0.08), 2),
        "customers": random.randint(int(orders * 0.7), orders),
        "sessions": random.randint(orders * 10, orders * 50),
        "conversion_rate": round(random.uniform(1, 5), 2),
        "mock_data": True
    }

def generate_woocommerce_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock WooCommerce data"""
    orders = random.randint(5, 80)
    revenue = round(orders * random.uniform(40, 180), 2)
    
    return {
        "id": f"woocommerce_{campaign_id}_{date.isoformat()}",
        "store_id": campaign_id,
        "store_name": campaign_name,
        "date": date.isoformat(),
        "orders": orders,
        "revenue": revenue,
        "average_order_value": round(revenue / orders, 2) if orders > 0 else 0,
        "products_sold": random.randint(orders, orders * 3),
        "refunds": round(random.uniform(0, revenue * 0.03), 2),
        "coupons_used": random.randint(0, int(orders * 0.2)),
        "new_customers": random.randint(0, int(orders * 0.3)),
        "returning_customers": orders - random.randint(0, int(orders * 0.3)),
        "mock_data": True
    }

def generate_amazon_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Amazon Seller Central data"""
    units_sold = random.randint(10, 200)
    revenue = round(units_sold * random.uniform(25, 100), 2)
    
    return {
        "id": f"amazon_{campaign_id}_{date.isoformat()}",
        "seller_id": campaign_id,
        "marketplace": "US",
        "date": date.isoformat(),
        "units_sold": units_sold,
        "revenue": revenue,
        "average_selling_price": round(revenue / units_sold, 2) if units_sold > 0 else 0,
        "fees": round(revenue * random.uniform(0.1, 0.2), 2),
        "advertising_spend": round(random.uniform(0, revenue * 0.15), 2),
        "sessions": random.randint(units_sold * 20, units_sold * 100),
        "conversion_rate": round(random.uniform(5, 15), 2),
        "returns": random.randint(0, int(units_sold * 0.05)),
        "mock_data": True
    }

def generate_hubspot_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock HubSpot CRM data"""
    contacts = random.randint(50, 500)
    deals = random.randint(5, 50)
    
    return {
        "id": f"hubspot_{campaign_id}_{date.isoformat()}",
        "portal_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "contacts_created": random.randint(0, int(contacts * 0.1)),
        "deals_created": random.randint(0, int(deals * 0.2)),
        "deals_closed_won": random.randint(0, int(deals * 0.3)),
        "deals_closed_lost": random.randint(0, int(deals * 0.2)),
        "deal_value": round(random.uniform(1000, 50000), 2),
        "emails_sent": random.randint(100, 1000),
        "emails_opened": random.randint(20, 300),
        "website_visits": random.randint(500, 5000),
        "form_submissions": random.randint(5, 50),
        "mock_data": True
    }

def generate_salesforce_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Salesforce data"""
    opportunities = random.randint(10, 100)
    
    return {
        "id": f"salesforce_{campaign_id}_{date.isoformat()}",
        "org_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "opportunities_created": random.randint(0, int(opportunities * 0.2)),
        "opportunities_closed_won": random.randint(0, int(opportunities * 0.25)),
        "opportunities_closed_lost": random.randint(0, int(opportunities * 0.15)),
        "pipeline_value": round(random.uniform(10000, 500000), 2),
        "leads_created": random.randint(20, 200),
        "leads_converted": random.randint(5, 50),
        "accounts_created": random.randint(1, 20),
        "activities_logged": random.randint(50, 500),
        "mock_data": True
    }

def generate_stripe_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Stripe payment data"""
    charges = random.randint(20, 200)
    revenue = round(charges * random.uniform(30, 150), 2)
    
    return {
        "id": f"stripe_{campaign_id}_{date.isoformat()}",
        "account_id": campaign_id,
        "date": date.isoformat(),
        "charges": charges,
        "successful_charges": int(charges * random.uniform(0.95, 0.99)),
        "failed_charges": charges - int(charges * random.uniform(0.95, 0.99)),
        "revenue": revenue,
        "fees": round(revenue * 0.029 + charges * 0.30, 2),  # Stripe's typical fee
        "refunds": round(random.uniform(0, revenue * 0.02), 2),
        "disputes": random.randint(0, int(charges * 0.01)),
        "new_customers": random.randint(0, int(charges * 0.3)),
        "mock_data": True
    }

def generate_paypal_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock PayPal data"""
    transactions = random.randint(15, 150)
    revenue = round(transactions * random.uniform(25, 120), 2)
    
    return {
        "id": f"paypal_{campaign_id}_{date.isoformat()}",
        "merchant_id": campaign_id,
        "date": date.isoformat(),
        "transactions": transactions,
        "revenue": revenue,
        "fees": round(revenue * 0.034 + transactions * 0.49, 2),  # PayPal's typical fee
        "refunds": round(random.uniform(0, revenue * 0.03), 2),
        "chargebacks": random.randint(0, int(transactions * 0.005)),
        "currency": "USD",
        "mock_data": True
    }

def generate_tiktok_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock TikTok for Business data"""
    spend = random.uniform(50, 500)
    impressions = int(spend * random.uniform(2000, 3000))
    clicks = int(impressions * random.uniform(0.015, 0.04))
    
    return {
        "id": f"tiktok_{campaign_id}_{date.isoformat()}",
        "advertiser_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "spend": round(spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "video_views": int(impressions * random.uniform(0.1, 0.3)),
        "video_view_rate": round(random.uniform(10, 30), 2),
        "cpm": round((spend / impressions) * 1000, 2) if impressions > 0 else 0,
        "cpc": round(spend / clicks, 2) if clicks > 0 else 0,
        "ctr": round((clicks / impressions) * 100, 2) if impressions > 0 else 0,
        "conversions": random.randint(0, int(clicks * 0.03)),
        "mock_data": True
    }

def generate_snowflake_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Snowflake data warehouse metrics"""
    return {
        "id": f"snowflake_{campaign_id}_{date.isoformat()}",
        "warehouse_id": campaign_id,
        "warehouse_name": campaign_name,
        "date": date.isoformat(),
        "queries_executed": random.randint(100, 1000),
        "compute_credits_used": round(random.uniform(10, 100), 2),
        "storage_bytes": random.randint(1000000000, 10000000000),  # 1GB to 10GB
        "data_transfer_bytes": random.randint(100000000, 1000000000),  # 100MB to 1GB
        "average_query_time": round(random.uniform(1, 30), 2),
        "failed_queries": random.randint(0, 50),
        "mock_data": True
    }

def generate_databricks_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Databricks analytics data"""
    return {
        "id": f"databricks_{campaign_id}_{date.isoformat()}",
        "workspace_id": campaign_id,
        "workspace_name": campaign_name,
        "date": date.isoformat(),
        "jobs_executed": random.randint(10, 100),
        "compute_hours": round(random.uniform(5, 50), 2),
        "data_processed_gb": round(random.uniform(10, 1000), 2),
        "notebooks_created": random.randint(0, 10),
        "clusters_created": random.randint(0, 5),
        "ml_models_trained": random.randint(0, 3),
        "mock_data": True
    }

def generate_zoho_crm_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock Zoho CRM data"""
    return {
        "id": f"zoho_{campaign_id}_{date.isoformat()}",
        "org_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "leads_created": random.randint(5, 50),
        "contacts_created": random.randint(3, 30),
        "deals_created": random.randint(2, 20),
        "deals_won": random.randint(0, 10),
        "deal_value": round(random.uniform(1000, 25000), 2),
        "calls_made": random.randint(10, 100),
        "emails_sent": random.randint(20, 200),
        "tasks_completed": random.randint(15, 150),
        "mock_data": True
    }

def generate_linkedin_ads_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock LinkedIn Ads data"""
    spend = random.uniform(100, 800)
    impressions = int(spend * random.uniform(500, 1000))
    clicks = int(impressions * random.uniform(0.005, 0.02))
    
    return {
        "id": f"linkedin_{campaign_id}_{date.isoformat()}",
        "account_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "spend": round(spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "cpm": round((spend / impressions) * 1000, 2) if impressions > 0 else 0,
        "cpc": round(spend / clicks, 2) if clicks > 0 else 0,
        "ctr": round((clicks / impressions) * 100, 2) if impressions > 0 else 0,
        "leads": random.randint(0, int(clicks * 0.1)),
        "video_views": int(impressions * random.uniform(0.05, 0.15)),
        "follows": random.randint(0, int(clicks * 0.05)),
        "mock_data": True
    }

def generate_x_ads_data(campaign_id: str, campaign_name: str, date: date) -> Dict[str, Any]:
    """Generate mock X (Twitter) Ads data"""
    spend = random.uniform(50, 400)
    impressions = int(spend * random.uniform(1500, 2500))
    clicks = int(impressions * random.uniform(0.01, 0.03))
    
    return {
        "id": f"x_{campaign_id}_{date.isoformat()}",
        "account_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "spend": round(spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "engagements": int(impressions * random.uniform(0.02, 0.05)),
        "retweets": random.randint(0, int(clicks * 0.1)),
        "likes": random.randint(0, int(clicks * 0.3)),
        "replies": random.randint(0, int(clicks * 0.05)),
        "follows": random.randint(0, int(clicks * 0.02)),
        "cpm": round((spend / impressions) * 1000, 2) if impressions > 0 else 0,
        "cpc": round(spend / clicks, 2) if clicks > 0 else 0,
        "mock_data": True
    }

def generate_generic_data(campaign_id: str, campaign_name: str, date: date, platform: str) -> Dict[str, Any]:
    """Generate generic mock data for unknown platforms"""
    return {
        "id": f"{platform}_{campaign_id}_{date.isoformat()}",
        "platform": platform,
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "date": date.isoformat(),
        "metric_1": round(random.uniform(100, 1000), 2),
        "metric_2": random.randint(1000, 10000),
        "metric_3": round(random.uniform(0.01, 0.1), 4),
        "mock_data": True
    }


class MockDataGenerator:
    """Class-based interface for generating mock marketing data"""
    
    def __init__(self):
        """Initialize the mock data generator"""
        # Map DataSource enum values to generator functions
        self.platform_generators = {
            DataSource.META_BUSINESS: generate_meta_business_data,
            DataSource.GOOGLE_ADS: generate_google_ads_data,
            DataSource.KLAVIYO: generate_klaviyo_data,
            DataSource.SHOPIFY: generate_shopify_data,
            DataSource.WOOCOMMERCE: generate_woocommerce_data,
            DataSource.AMAZON_SELLER_CENTRAL: generate_amazon_data,
            DataSource.HUBSPOT: generate_hubspot_data,
            DataSource.SALESFORCE: generate_salesforce_data,
            DataSource.STRIPE: generate_stripe_data,
            DataSource.PAYPAL: generate_paypal_data,
            DataSource.TIKTOK: generate_tiktok_data,
            DataSource.SNOWFLAKE: generate_snowflake_data,
            DataSource.DATABRICKS: generate_databricks_data,
            DataSource.ZOHO_CRM: generate_zoho_crm_data,
            DataSource.LINKEDIN_ADS: generate_linkedin_ads_data,
            DataSource.X_ADS: generate_x_ads_data,
        }
    
    def generate_platform_data(self, data_source: DataSource, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Generate mock data for a specific platform and date range
        
        Args:
            data_source: The platform data source
            start_date: Start date for data generation
            end_date: End date for data generation
            
        Returns:
            List of mock data records
        """
        # Get the appropriate generator function
        generator_func = self.platform_generators.get(data_source)
        
        if not generator_func:
            # Fall back to generic data generation
            return self._generate_generic_platform_data(data_source, start_date, end_date)
        
        # Generate data for each day in the range
        data_records = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate a few campaigns per day
            num_campaigns = random.randint(1, 3)
            
            for i in range(num_campaigns):
                campaign_id = f"campaign_{data_source.value}_{current_date.strftime('%Y%m%d')}_{i+1}"
                campaign_name = f"{data_source.value.replace('_', ' ').title()} Campaign {i+1}"
                
                # Generate data for this campaign and date
                campaign_data = generator_func(campaign_id, campaign_name, current_date)
                data_records.append(campaign_data)
            
            current_date += timedelta(days=1)
        
        return data_records
    
    def _generate_generic_platform_data(self, data_source: DataSource, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Generate generic mock data for platforms without specific generators"""
        data_records = []
        current_date = start_date
        
        while current_date <= end_date:
            campaign_id = f"campaign_{data_source.value}_{current_date.strftime('%Y%m%d')}"
            campaign_name = f"{data_source.value.replace('_', ' ').title()} Campaign"
            
            # Use the generic data generator
            campaign_data = generate_generic_data(campaign_id, campaign_name, current_date, data_source.value)
            data_records.append(campaign_data)
            
            current_date += timedelta(days=1)
        
        return data_records