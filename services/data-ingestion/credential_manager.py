"""
Credential Management for Data Ingestion Service
Handles secure storage and retrieval of API credentials
"""
import os
import json
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CredentialProvider(Enum):
    """Supported credential providers"""
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"  # Future implementation

class CredentialManager:
    """Manages API credentials for external platforms"""
    
    def __init__(self, provider: CredentialProvider = CredentialProvider.ENVIRONMENT):
        self.provider = provider
        self._credentials_cache = {}
    
    async def get_meta_business_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Meta Business API credentials"""
        cache_key = f"meta_business_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_meta_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_meta_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Meta Business credentials for org {org_id}")
        else:
            logger.warning(f"No Meta Business credentials found for org {org_id}")
        
        return credentials
    
    async def get_google_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Google Ads API credentials"""
        cache_key = f"google_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_google_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_google_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Google Ads credentials for org {org_id}")
        else:
            logger.warning(f"No Google Ads credentials found for org {org_id}")
        
        return credentials
    
    async def get_klaviyo_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Klaviyo API credentials"""
        cache_key = f"klaviyo_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_klaviyo_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_klaviyo_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Klaviyo credentials for org {org_id}")
        else:
            logger.warning(f"No Klaviyo credentials found for org {org_id}")
        
        return credentials
    
    # Tier 1 Platform Credentials
    async def get_shopify_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Shopify API credentials"""
        cache_key = f"shopify_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_shopify_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_shopify_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Shopify credentials for org {org_id}")
        else:
            logger.warning(f"No Shopify credentials found for org {org_id}")
        
        return credentials
    
    async def get_woocommerce_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get WooCommerce API credentials"""
        cache_key = f"woocommerce_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_woocommerce_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_woocommerce_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved WooCommerce credentials for org {org_id}")
        else:
            logger.warning(f"No WooCommerce credentials found for org {org_id}")
        
        return credentials
    
    async def get_amazon_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Amazon Seller Central API credentials"""
        cache_key = f"amazon_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_amazon_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_amazon_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Amazon credentials for org {org_id}")
        else:
            logger.warning(f"No Amazon credentials found for org {org_id}")
        
        return credentials
    
    # Tier 2 Platform Credentials
    async def get_hubspot_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get HubSpot CRM API credentials"""
        cache_key = f"hubspot_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_hubspot_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_hubspot_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved HubSpot credentials for org {org_id}")
        else:
            logger.warning(f"No HubSpot credentials found for org {org_id}")
        
        return credentials
    
    async def get_salesforce_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Salesforce CRM API credentials"""
        cache_key = f"salesforce_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_salesforce_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_salesforce_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Salesforce credentials for org {org_id}")
        else:
            logger.warning(f"No Salesforce credentials found for org {org_id}")
        
        return credentials
    
    async def get_stripe_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Stripe Payment API credentials"""
        cache_key = f"stripe_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_stripe_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_stripe_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Stripe credentials for org {org_id}")
        else:
            logger.warning(f"No Stripe credentials found for org {org_id}")
        
        return credentials
    
    async def get_paypal_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get PayPal Payment API credentials"""
        cache_key = f"paypal_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_paypal_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_paypal_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved PayPal credentials for org {org_id}")
        else:
            logger.warning(f"No PayPal credentials found for org {org_id}")
        
        return credentials
    
    async def get_tiktok_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get TikTok for Business API credentials"""
        cache_key = f"tiktok_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_tiktok_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_tiktok_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved TikTok credentials for org {org_id}")
        else:
            logger.warning(f"No TikTok credentials found for org {org_id}")
        
        return credentials
    
    async def get_snowflake_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Snowflake Data Warehouse credentials"""
        cache_key = f"snowflake_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_snowflake_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_snowflake_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Snowflake credentials for org {org_id}")
        else:
            logger.warning(f"No Snowflake credentials found for org {org_id}")
        
        return credentials
    
    async def get_databricks_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Databricks Analytics Platform credentials"""
        cache_key = f"databricks_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_databricks_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_databricks_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Databricks credentials for org {org_id}")
        else:
            logger.warning(f"No Databricks credentials found for org {org_id}")
        
        return credentials
    
    # Tier 4 Platform Credentials (Extended Social/CRM)
    async def get_zoho_crm_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Zoho CRM API credentials"""
        cache_key = f"zoho_crm_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_zoho_crm_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_zoho_crm_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Zoho CRM credentials for org {org_id}")
        else:
            logger.warning(f"No Zoho CRM credentials found for org {org_id}")
        
        return credentials
    
    async def get_linkedin_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get LinkedIn Ads API credentials"""
        cache_key = f"linkedin_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_linkedin_ads_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_linkedin_ads_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved LinkedIn Ads credentials for org {org_id}")
        else:
            logger.warning(f"No LinkedIn Ads credentials found for org {org_id}")
        
        return credentials
    
    async def get_x_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get X (Twitter) Ads API credentials"""
        cache_key = f"x_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_x_ads_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_x_ads_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved X Ads credentials for org {org_id}")
        else:
            logger.warning(f"No X Ads credentials found for org {org_id}")
        
        return credentials
    
    def _get_meta_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Meta Business credentials from environment variables"""
        access_token = os.getenv("META_ACCESS_TOKEN")
        app_id = os.getenv("META_APP_ID")
        app_secret = os.getenv("META_APP_SECRET")
        
        if access_token and app_id and app_secret:
            return {
                "access_token": access_token,
                "app_id": app_id,
                "app_secret": app_secret
            }
        return None
    
    def _get_google_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Google Ads credentials from environment variables"""
        developer_token = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
        client_id = os.getenv("GOOGLE_ADS_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_ADS_CLIENT_SECRET")
        refresh_token = os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
        
        if developer_token and client_id and client_secret and refresh_token:
            return {
                "developer_token": developer_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token
            }
        return None
    
    def _get_klaviyo_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Klaviyo credentials from environment variables"""
        api_key = os.getenv("KLAVIYO_API_KEY")
        
        if api_key:
            return {
                "api_key": api_key
            }
        return None
    
    # Tier 1 Environment Credential Methods
    def _get_shopify_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Shopify credentials from environment variables"""
        shop_domain = os.getenv("SHOPIFY_SHOP_DOMAIN")
        access_token = os.getenv("SHOPIFY_ACCESS_TOKEN")
        
        if shop_domain and access_token:
            return {
                "shop_domain": shop_domain,
                "access_token": access_token
            }
        return None
    
    def _get_woocommerce_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get WooCommerce credentials from environment variables"""
        site_url = os.getenv("WOOCOMMERCE_SITE_URL")
        consumer_key = os.getenv("WOOCOMMERCE_CONSUMER_KEY")
        consumer_secret = os.getenv("WOOCOMMERCE_CONSUMER_SECRET")
        
        if site_url and consumer_key and consumer_secret:
            return {
                "site_url": site_url,
                "consumer_key": consumer_key,
                "consumer_secret": consumer_secret
            }
        return None
    
    def _get_amazon_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Amazon Seller Central credentials from environment variables"""
        marketplace_id = os.getenv("AMAZON_MARKETPLACE_ID")
        seller_id = os.getenv("AMAZON_SELLER_ID")
        aws_access_key = os.getenv("AMAZON_AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AMAZON_AWS_SECRET_KEY")
        role_arn = os.getenv("AMAZON_ROLE_ARN")
        client_id = os.getenv("AMAZON_CLIENT_ID")
        client_secret = os.getenv("AMAZON_CLIENT_SECRET")
        refresh_token = os.getenv("AMAZON_REFRESH_TOKEN")
        
        if all([marketplace_id, seller_id, aws_access_key, aws_secret_key,
                role_arn, client_id, client_secret, refresh_token]):
            return {
                "marketplace_id": marketplace_id,
                "seller_id": seller_id,
                "aws_access_key": aws_access_key,
                "aws_secret_key": aws_secret_key,
                "role_arn": role_arn,
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token
            }
        return None
    
    # Tier 2 Environment Credential Methods
    def _get_hubspot_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get HubSpot credentials from environment variables"""
        api_key = os.getenv("HUBSPOT_API_KEY")
        
        if api_key:
            return {
                "api_key": api_key
            }
        return None
    
    def _get_salesforce_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Salesforce credentials from environment variables"""
        username = os.getenv("SALESFORCE_USERNAME")
        password = os.getenv("SALESFORCE_PASSWORD")
        security_token = os.getenv("SALESFORCE_SECURITY_TOKEN")
        client_id = os.getenv("SALESFORCE_CLIENT_ID")
        client_secret = os.getenv("SALESFORCE_CLIENT_SECRET")
        is_sandbox = os.getenv("SALESFORCE_IS_SANDBOX", "false")
        
        if all([username, password, security_token, client_id, client_secret]):
            return {
                "username": username,
                "password": password,
                "security_token": security_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "is_sandbox": is_sandbox
            }
        return None
    
    def _get_stripe_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Stripe credentials from environment variables"""
        api_key = os.getenv("STRIPE_API_KEY")
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        if api_key:
            credentials = {"api_key": api_key}
            if webhook_secret:
                credentials["webhook_secret"] = webhook_secret
            return credentials
        return None
    
    def _get_paypal_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get PayPal credentials from environment variables"""
        client_id = os.getenv("PAYPAL_CLIENT_ID")
        client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
        is_sandbox = os.getenv("PAYPAL_IS_SANDBOX", "false")
        webhook_id = os.getenv("PAYPAL_WEBHOOK_ID")
        
        if client_id and client_secret:
            credentials = {
                "client_id": client_id,
                "client_secret": client_secret,
                "is_sandbox": is_sandbox
            }
            if webhook_id:
                credentials["webhook_id"] = webhook_id
            return credentials
        return None
    
    def _get_tiktok_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get TikTok for Business credentials from environment variables"""
        access_token = os.getenv("TIKTOK_ACCESS_TOKEN")
        app_id = os.getenv("TIKTOK_APP_ID")
        app_secret = os.getenv("TIKTOK_APP_SECRET")
        
        if access_token and app_id and app_secret:
            return {
                "access_token": access_token,
                "app_id": app_id,
                "app_secret": app_secret
            }
        return None
    
    def _get_snowflake_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Snowflake credentials from environment variables"""
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        username = os.getenv("SNOWFLAKE_USERNAME")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA")
        
        if account and username and password and warehouse and database and schema:
            return {
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema
            }
        return None
    
    def _get_databricks_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Databricks credentials from environment variables"""
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")
        
        if host and token and cluster_id:
            return {
                "host": host,
                "token": token,
                "cluster_id": cluster_id
            }
        return None
    
    # Tier 4 Environment Credential Methods
    def _get_zoho_crm_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Zoho CRM credentials from environment variables"""
        client_id = os.getenv("ZOHO_CRM_CLIENT_ID")
        client_secret = os.getenv("ZOHO_CRM_CLIENT_SECRET")
        refresh_token = os.getenv("ZOHO_CRM_REFRESH_TOKEN")
        domain = os.getenv("ZOHO_CRM_DOMAIN", "com")  # Default to .com
        
        if client_id and client_secret and refresh_token:
            return {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "domain": domain
            }
        return None
    
    def _get_linkedin_ads_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get LinkedIn Ads credentials from environment variables"""
        client_id = os.getenv("LINKEDIN_ADS_CLIENT_ID")
        client_secret = os.getenv("LINKEDIN_ADS_CLIENT_SECRET")
        access_token = os.getenv("LINKEDIN_ADS_ACCESS_TOKEN")
        
        if client_id and client_secret and access_token:
            return {
                "client_id": client_id,
                "client_secret": client_secret,
                "access_token": access_token
            }
        return None
    
    def _get_x_ads_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get X (Twitter) Ads credentials from environment variables"""
        consumer_key = os.getenv("X_ADS_CONSUMER_KEY")
        consumer_secret = os.getenv("X_ADS_CONSUMER_SECRET")
        access_token = os.getenv("X_ADS_ACCESS_TOKEN")
        access_token_secret = os.getenv("X_ADS_ACCESS_TOKEN_SECRET")
        
        if consumer_key and consumer_secret and access_token and access_token_secret:
            return {
                "consumer_key": consumer_key,
                "consumer_secret": consumer_secret,
                "access_token": access_token,
                "access_token_secret": access_token_secret
            }
        return None
    
    def _get_meta_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Meta Business credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/meta_business.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_google_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Google Ads credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/google_ads.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_klaviyo_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Klaviyo credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/klaviyo.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_shopify_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Shopify credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/shopify.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_woocommerce_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get WooCommerce credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/woocommerce.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_amazon_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Amazon Seller Central credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/amazon.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_hubspot_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get HubSpot credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/hubspot.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_salesforce_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Salesforce credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/salesforce.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_stripe_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Stripe credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/stripe.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_paypal_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get PayPal credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/paypal.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_tiktok_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get TikTok credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/tiktok.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_snowflake_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Snowflake credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/snowflake.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_databricks_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Databricks credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/databricks.json"
        return self._load_credentials_from_file(credentials_file)
    
    # Tier 4 File Credential Methods
    def _get_zoho_crm_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Zoho CRM credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/zoho_crm.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_linkedin_ads_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get LinkedIn Ads credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/linkedin_ads.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_x_ads_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get X Ads credentials from file"""
        credentials_file = f"/app/credentials/{org_id}/x_ads.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _load_credentials_from_file(self, file_path: str) -> Optional[Dict[str, str]]:
        """Load credentials from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials from {file_path}: {str(e)}")
        return None
    
    def clear_cache(self):
        """Clear credentials cache"""
        self._credentials_cache.clear()
        logger.info("Credentials cache cleared")
    
    def validate_credentials(self, platform: str, credentials: Dict[str, str]) -> bool:
        """Validate that credentials contain required fields"""
        required_fields = {
            # Tier 0 (Legacy)
            "meta_business": ["access_token", "app_id", "app_secret"],
            "google_ads": ["developer_token", "client_id", "client_secret", "refresh_token"],
            "klaviyo": ["api_key"],
            
            # Tier 1 (E-commerce)
            "shopify": ["shop_domain", "access_token"],
            "woocommerce": ["base_url", "consumer_key", "consumer_secret"],
            "amazon": ["access_key", "secret_key", "marketplace_id"],
            
            # Tier 2 (CRM/Payment)
            "hubspot": ["access_token"],
            "salesforce": ["username", "password", "security_token", "domain"],
            "stripe": ["secret_key"],
            "paypal": ["client_id", "client_secret"],
            
            # Tier 3 (Social/Analytics/Data)
            "tiktok": ["access_token", "app_id", "app_secret"],
            "snowflake": ["account", "username", "password", "warehouse", "database", "schema"],
            "databricks": ["host", "token", "cluster_id"],
            
            # Tier 4 (Extended Social/CRM)
            "zoho_crm": ["client_id", "client_secret", "refresh_token", "domain"],
            "linkedin_ads": ["client_id", "client_secret", "access_token"],
            "x_ads": ["consumer_key", "consumer_secret", "access_token", "access_token_secret"]
        }
        
        if platform not in required_fields:
            return False
        
        return all(field in credentials for field in required_fields[platform])

# Global credential manager instance
credential_manager = CredentialManager()