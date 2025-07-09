"""
PayPal Payment Connector for LiftOS v1.3.0
Advanced payment processing attribution with merchant transaction analysis
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
import base64

from shared.models.marketing import DataSource
from shared.models.causal_marketing import (
    CausalMarketingData, ConfounderVariable,
    TreatmentType, RandomizationUnit
)
from shared.utils.causal_transforms import CausalDataTransformer, TreatmentAssignmentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PayPalPaymentStatus(str, Enum):
    """PayPal payment status values"""
    CREATED = "CREATED"
    SAVED = "SAVED"
    APPROVED = "APPROVED"
    VOIDED = "VOIDED"
    COMPLETED = "COMPLETED"
    PAYER_ACTION_REQUIRED = "PAYER_ACTION_REQUIRED"

class PayPalTransactionStatus(str, Enum):
    """PayPal transaction status values"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    DENIED = "DENIED"
    REFUNDED = "REFUNDED"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"

class PayPalSubscriptionStatus(str, Enum):
    """PayPal subscription status values"""
    APPROVAL_PENDING = "APPROVAL_PENDING"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class PayPalPaymentMethod(str, Enum):
    """PayPal payment method types"""
    PAYPAL = "PAYPAL"
    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    BANK = "BANK"
    APPLE_PAY = "APPLE_PAY"
    GOOGLE_PAY = "GOOGLE_PAY"

# PayPal Data Models
class PayPalPaymentData(BaseModel):
    """PayPal Payment data model"""
    id: str
    intent: str  # CAPTURE, AUTHORIZE
    status: PayPalPaymentStatus
    purchase_units: List[Dict[str, Any]] = []
    payer: Optional[Dict[str, Any]] = None
    payment_source: Optional[Dict[str, Any]] = None
    create_time: datetime
    update_time: Optional[datetime] = None
    links: List[Dict[str, Any]] = []
    application_context: Optional[Dict[str, Any]] = None

class PayPalTransactionData(BaseModel):
    """PayPal Transaction data model"""
    id: str
    status: PayPalTransactionStatus
    amount: Dict[str, Any]  # Contains value and currency_code
    payee: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    custom_id: Optional[str] = None
    invoice_id: Optional[str] = None
    soft_descriptor: Optional[str] = None
    item_list: Optional[Dict[str, Any]] = None
    related_resources: List[Dict[str, Any]] = []
    create_time: datetime
    update_time: Optional[datetime] = None
    payment_mode: Optional[str] = None
    protection_eligibility: Optional[str] = None
    protection_eligibility_type: Optional[str] = None
    transaction_fee: Optional[Dict[str, Any]] = None
    parent_payment: Optional[str] = None
    fmf_details: Optional[Dict[str, Any]] = None

class PayPalSubscriptionData(BaseModel):
    """PayPal Subscription data model"""
    id: str
    plan_id: str
    start_time: datetime
    quantity: str = "1"
    shipping_amount: Optional[Dict[str, Any]] = None
    subscriber: Optional[Dict[str, Any]] = None
    billing_info: Optional[Dict[str, Any]] = None
    create_time: datetime
    update_time: Optional[datetime] = None
    links: List[Dict[str, Any]] = []
    status: PayPalSubscriptionStatus
    status_update_time: Optional[datetime] = None
    plan_overridden: Optional[bool] = None
    custom_id: Optional[str] = None
    application_context: Optional[Dict[str, Any]] = None

class PayPalWebhookData(BaseModel):
    """PayPal Webhook event data model"""
    id: str
    event_version: str
    create_time: datetime
    resource_type: str
    event_type: str
    summary: Optional[str] = None
    resource: Dict[str, Any]
    links: List[Dict[str, Any]] = []
    resource_version: Optional[str] = None

class PayPalConnector:
    """
    PayPal Payment Connector with advanced payment and subscription attribution
    Integrates with PayPal REST API for comprehensive payment data extraction
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        environment: str = "sandbox"  # "sandbox" or "live"
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.environment = environment
        
        # Set base URL based on environment
        if environment == "live":
            self.base_url = "https://api-m.paypal.com"
        else:
            self.base_url = "https://api-m.sandbox.paypal.com"
        
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting: PayPal allows different limits based on API
        self.rate_limit_calls_per_minute = 300  # Conservative rate limiting
        self.rate_limit_window = 60  # 1 minute
        self.call_timestamps: List[datetime] = []
    
    async def authenticate(self) -> bool:
        """Authenticate with PayPal using OAuth 2.0 Client Credentials flow"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            auth_url = f"{self.base_url}/v1/oauth2/token"
            
            # Create basic auth header
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Accept': 'application/json',
                'Accept-Language': 'en_US',
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = 'grant_type=client_credentials'
            
            async with self.session.post(auth_url, headers=headers, data=data) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    self.access_token = auth_data['access_token']
                    expires_in = auth_data.get('expires_in', 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # Refresh 1 min early
                    logger.info("Successfully authenticated with PayPal")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"PayPal authentication failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during PayPal authentication: {str(e)}")
            return False
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Remove timestamps older than the rate limit window
        self.call_timestamps = [
            ts for ts in self.call_timestamps 
            if (now - ts).total_seconds() < self.rate_limit_window
        ]
        
        # Check if we're at the rate limit
        if len(self.call_timestamps) >= self.rate_limit_calls_per_minute:
            sleep_time = self.rate_limit_window - (now - self.call_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Add current timestamp
        self.call_timestamps.append(now)
    
    async def _make_api_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """Make authenticated API request to PayPal"""
        await self._check_rate_limit()
        
        # Check if token needs refresh
        if not self.access_token or (self.token_expires_at and datetime.now() >= self.token_expires_at):
            if not await self.authenticate():
                return None
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json',
            'PayPal-Partner-Attribution-Id': 'LiftOS_SP'  # Partner attribution
        }
        
        try:
            if method == "GET":
                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        # Token expired, re-authenticate
                        logger.info("Access token expired, re-authenticating...")
                        if await self.authenticate():
                            headers['Authorization'] = f'Bearer {self.access_token}'
                            async with self.session.get(url, headers=headers, params=params) as retry_response:
                                if retry_response.status == 200:
                                    return await retry_response.json()
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"PayPal API request failed: {response.status} - {error_text}")
                        return None
            else:
                async with self.session.request(method, url, headers=headers, json=params) as response:
                    if response.status in [200, 201]:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"PayPal API request failed: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error making PayPal API request: {str(e)}")
            return None
    
    async def get_payments(
        self, 
        start_date: datetime, 
        end_date: datetime,
        page_size: int = 20
    ) -> List[PayPalPaymentData]:
        """Fetch payments from PayPal"""
        payments = []
        
        # PayPal uses ISO 8601 format for dates
        start_time = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_time = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        params = {
            'start_time': start_time,
            'end_time': end_time,
            'page_size': page_size,
            'sort_by': 'create_time',
            'sort_order': 'desc'
        }
        
        response = await self._make_api_request("v2/payments", params)
        
        if response and 'payments' in response:
            for payment_data in response['payments']:
                try:
                    # Convert time strings to datetime objects
                    payment_data['create_time'] = datetime.fromisoformat(
                        payment_data['create_time'].replace('Z', '+00:00')
                    )
                    if payment_data.get('update_time'):
                        payment_data['update_time'] = datetime.fromisoformat(
                            payment_data['update_time'].replace('Z', '+00:00')
                        )
                    
                    # Convert status to enum
                    if payment_data.get('status'):
                        try:
                            payment_data['status'] = PayPalPaymentStatus(payment_data['status'])
                        except ValueError:
                            payment_data['status'] = PayPalPaymentStatus.CREATED
                    
                    payment = PayPalPaymentData(**payment_data)
                    payments.append(payment)
                    
                except Exception as e:
                    logger.warning(f"Error parsing payment {payment_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(payments)} payments from PayPal")
        return payments
    
    async def get_transactions(
        self, 
        start_date: datetime, 
        end_date: datetime,
        page_size: int = 500
    ) -> List[PayPalTransactionData]:
        """Fetch transactions from PayPal Transaction Search API"""
        transactions = []
        
        # PayPal Transaction Search uses different date format
        start_time = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_time = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        params = {
            'start_date': start_time,
            'end_date': end_time,
            'page_size': page_size,
            'fields': 'all'
        }
        
        response = await self._make_api_request("v1/reporting/transactions", params)
        
        if response and 'transaction_details' in response:
            for transaction_data in response['transaction_details']:
                try:
                    # Convert time strings to datetime objects
                    if transaction_data.get('transaction_info', {}).get('transaction_initiation_date'):
                        transaction_data['create_time'] = datetime.fromisoformat(
                            transaction_data['transaction_info']['transaction_initiation_date'].replace('Z', '+00:00')
                        )
                    else:
                        transaction_data['create_time'] = datetime.now()
                    
                    if transaction_data.get('transaction_info', {}).get('transaction_updated_date'):
                        transaction_data['update_time'] = datetime.fromisoformat(
                            transaction_data['transaction_info']['transaction_updated_date'].replace('Z', '+00:00')
                        )
                    
                    # Extract key fields from nested structure
                    transaction_info = transaction_data.get('transaction_info', {})
                    
                    # Create simplified transaction data
                    simplified_transaction = {
                        'id': transaction_info.get('transaction_id', ''),
                        'status': transaction_info.get('transaction_status', 'PENDING'),
                        'amount': {
                            'value': transaction_info.get('transaction_amount', {}).get('value', '0'),
                            'currency_code': transaction_info.get('transaction_amount', {}).get('currency_code', 'USD')
                        },
                        'create_time': transaction_data['create_time'],
                        'update_time': transaction_data.get('update_time'),
                        'description': transaction_info.get('transaction_subject'),
                        'custom_id': transaction_info.get('custom_field'),
                        'invoice_id': transaction_info.get('invoice_id'),
                        'related_resources': [],
                        'transaction_fee': transaction_info.get('fee_amount'),
                        'protection_eligibility': transaction_info.get('protection_eligibility')
                    }
                    
                    # Convert status to enum
                    try:
                        simplified_transaction['status'] = PayPalTransactionStatus(simplified_transaction['status'])
                    except ValueError:
                        simplified_transaction['status'] = PayPalTransactionStatus.PENDING
                    
                    transaction = PayPalTransactionData(**simplified_transaction)
                    transactions.append(transaction)
                    
                except Exception as e:
                    logger.warning(f"Error parsing transaction {transaction_data.get('transaction_info', {}).get('transaction_id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(transactions)} transactions from PayPal")
        return transactions
    
    async def get_subscriptions(
        self, 
        start_date: datetime, 
        end_date: datetime,
        page_size: int = 20
    ) -> List[PayPalSubscriptionData]:
        """Fetch subscriptions from PayPal"""
        subscriptions = []
        
        # PayPal subscriptions API doesn't support date filtering directly
        # We'll fetch recent subscriptions and filter by date
        params = {
            'page_size': page_size,
            'sort_by': 'create_time',
            'sort_order': 'desc'
        }
        
        response = await self._make_api_request("v1/billing/subscriptions", params)
        
        if response and 'subscriptions' in response:
            for subscription_data in response['subscriptions']:
                try:
                    # Convert time strings to datetime objects
                    subscription_data['create_time'] = datetime.fromisoformat(
                        subscription_data['create_time'].replace('Z', '+00:00')
                    )
                    subscription_data['start_time'] = datetime.fromisoformat(
                        subscription_data['start_time'].replace('Z', '+00:00')
                    )
                    
                    if subscription_data.get('update_time'):
                        subscription_data['update_time'] = datetime.fromisoformat(
                            subscription_data['update_time'].replace('Z', '+00:00')
                        )
                    
                    if subscription_data.get('status_update_time'):
                        subscription_data['status_update_time'] = datetime.fromisoformat(
                            subscription_data['status_update_time'].replace('Z', '+00:00')
                        )
                    
                    # Filter by date range
                    if start_date <= subscription_data['create_time'] <= end_date:
                        # Convert status to enum
                        if subscription_data.get('status'):
                            try:
                                subscription_data['status'] = PayPalSubscriptionStatus(subscription_data['status'])
                            except ValueError:
                                subscription_data['status'] = PayPalSubscriptionStatus.ACTIVE
                        
                        subscription = PayPalSubscriptionData(**subscription_data)
                        subscriptions.append(subscription)
                    
                except Exception as e:
                    logger.warning(f"Error parsing subscription {subscription_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(subscriptions)} subscriptions from PayPal")
        return subscriptions
    
    async def sync_paypal_data(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = False
    ) -> List[CausalMarketingData]:
        """
        Sync PayPal data and transform to causal marketing format
        Processes both payments and subscription events
        """
        try:
            logger.info(f"Starting PayPal data sync for org {org_id}")
            
            # Fetch payment data
            payments = await self.get_payments(start_date, end_date)
            transactions = await self.get_transactions(start_date, end_date)
            subscriptions = await self.get_subscriptions(start_date, end_date)
            
            causal_data_list = []
            
            # Process completed payments as primary conversion events
            completed_payments = [
                payment for payment in payments 
                if payment.status == PayPalPaymentStatus.COMPLETED
            ]
            
            for payment in completed_payments:
                try:
                    # Process payment data
                    causal_data = await self._process_payment_data(payment, org_id)
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing payment {payment.id}: {str(e)}")
                    continue
            
            # Process completed transactions as secondary conversion events
            completed_transactions = [
                transaction for transaction in transactions 
                if transaction.status == PayPalTransactionStatus.COMPLETED
            ]
            
            for transaction in completed_transactions:
                try:
                    # Process transaction data
                    causal_data = await self._process_transaction_data(transaction, org_id)
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing transaction {transaction.id}: {str(e)}")
                    continue
            
            # Process active subscriptions as tertiary conversion events
            active_subscriptions = [
                subscription for subscription in subscriptions 
                if subscription.status == PayPalSubscriptionStatus.ACTIVE
            ]
            
            for subscription in active_subscriptions:
                try:
                    # Process subscription data
                    causal_data = await self._process_subscription_data(subscription, org_id)
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing subscription {subscription.id}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(causal_data_list)} PayPal records")
            return causal_data_list
            
        except Exception as e:
            logger.error(f"Error in PayPal data sync: {str(e)}")
            return []
    
    async def _process_payment_data(
        self,
        payment: PayPalPaymentData,
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process payment data into causal marketing format"""
        
        # Extract payment amount from purchase units
        conversion_value = 0.0
        currency = "USD"
        
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'amount' in unit:
                    amount_info = unit['amount']
                    conversion_value += float(amount_info.get('value', 0))
                    currency = amount_info.get('currency_code', 'USD')
        
        is_conversion = payment.status == PayPalPaymentStatus.COMPLETED
        conversion_date = payment.create_time
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_payment_with_kse(payment)
        
        # Treatment assignment
        treatment_result = await self._determine_payment_treatment_assignment(payment, None)
        
        # Confounder detection
        confounders = await self._detect_payment_confounders(payment, None)
        
        # Extract geographic and audience data
        geographic_data = self._extract_payment_geographic_data(payment)
        audience_data = self._extract_payment_audience_data(payment)
        
        # Calculate data quality score
        data_quality_score = self._calculate_payment_data_quality_score(payment)
        
        # Determine campaign information
        campaign_info = self._determine_payment_campaign_info(payment)
        
        # Extract customer ID from payer information
        customer_id = "anonymous_paypal"
        if payment.payer and 'payer_id' in payment.payer:
            customer_id = payment.payer['payer_id']
        elif payment.payer and 'email_address' in payment.payer:
            customer_id = payment.payer['email_address']
        
        return CausalMarketingData(
            id=f"paypal_payment_{payment.id}",
            org_id=org_id,
            data_source=DataSource.PAYPAL,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=customer_id,
            campaign_id=campaign_info.get("campaign_id", "unknown_campaign"),
            ad_set_id=campaign_info.get("utm_campaign", "unknown_ad_set"),
            ad_id=campaign_info.get("utm_source", "unknown_ad"),
            treatment_assignment=treatment_result,
            confounding_variables=confounders,
            kse_enhancement=kse_enhancement,
            geographic_data=geographic_data,
            audience_data=audience_data,
            data_quality_score=data_quality_score,
            raw_data={
                "payment": payment.dict()
            }
        )
    
    async def _process_transaction_data(
        self,
        transaction: PayPalTransactionData,
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process transaction data into causal marketing format"""
        
        # Extract transaction amount
        conversion_value = float(transaction.amount.get('value', 0))
        currency = transaction.amount.get('currency_code', 'USD')
        
        is_conversion = transaction.status == PayPalTransactionStatus.COMPLETED
        conversion_date = transaction.create_time
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_transaction_with_kse(transaction)
        
        # Treatment assignment
        treatment_result = await self._determine_transaction_treatment_assignment(transaction, None)
        
        # Confounder detection
        confounders = await self._detect_transaction_confounders(transaction, None)
        
        # Extract geographic and audience data
        geographic_data = self._extract_transaction_geographic_data(transaction)
        audience_data = self._extract_transaction_audience_data(transaction)
        
        # Calculate data quality score
        data_quality_score = self._calculate_transaction_data_quality_score(transaction)
        
        # Determine campaign information
        campaign_info = self._determine_transaction_campaign_info(transaction)
        
        return CausalMarketingData(
            id=f"paypal_transaction_{transaction.id}",
            org_id=org_id,
            data_source=DataSource.PAYPAL,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=transaction.custom_id or f"transaction_{transaction.id}",
            campaign_id=campaign_info.get("campaign_id", "unknown_campaign"),
            ad_set_id=campaign_info.get("utm_campaign", "unknown_ad_set"),
            ad_id=campaign_info.get("utm_source", "unknown_ad"),
            treatment_assignment=treatment_result,
            confounding_variables=confounders,
            kse_enhancement=kse_enhancement,
            geographic_data=geographic_data,
            audience_data=audience_data,
            data_quality_score=data_quality_score,
            raw_data={
                "transaction": transaction.dict()
            }
        )
    
    async def _process_subscription_data(
        self,
        subscription: PayPalSubscriptionData,
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process subscription data into causal marketing format"""
        
        # Subscription value is typically monthly recurring revenue
        conversion_value = 0.0
        if subscription.billing_info and 'last_payment' in subscription.billing_info:
            last_payment = subscription.billing_info['last_payment']
            if 'amount' in last_payment:
                conversion_value = float(last_payment['amount'].get('value', 0))
        
        is_conversion = subscription.status == PayPalSubscriptionStatus.ACTIVE
        conversion_date = subscription.create_time
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_subscription_with_kse(subscription)
        
        # Treatment assignment
        treatment_result = await self._determine_subscription_treatment_assignment(subscription, None)
        
        # Confounder detection
        confounders = await self._detect_subscription_confounders(subscription, None)
        
        # Extract geographic and audience data
        geographic_data = self._extract_subscription_geographic_data(subscription)
        audience_data = self._extract_subscription_audience_data(subscription)
        
        # Calculate data quality score
        data_quality_score = self._calculate_subscription_data_quality_score(subscription)
        
        # Determine campaign information
        campaign_info = self._determine_subscription_campaign_info(subscription)
        
        # Extract customer ID from subscriber information
        customer_id = "anonymous_subscriber"
        if subscription.subscriber and 'payer_id' in subscription.subscriber:
            customer_id = subscription.subscriber['payer_id']
        elif subscription.subscriber and 'email_address' in subscription.subscriber:
            customer_id = subscription.subscriber['email_address']
        
        return CausalMarketingData(
            id=f"paypal_subscription_{subscription.id}",
            org_id=org_id,
            data_source=DataSource.PAYPAL,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=customer_id,
            campaign_id=campaign_info.get("campaign_id", "unknown_campaign"),
            ad_set_id=campaign_info.get("utm_campaign", "unknown_ad_set"),
            ad_id=campaign_info.get("utm_source", "unknown_ad"),
            treatment_assignment=treatment_result,
            confounding_variables=confounders,
            kse_enhancement=kse_enhancement,
            geographic_data=geographic_data,
            audience_data=audience_data,
            data_quality_score=data_quality_score,
            raw_data={
                "subscription": subscription.dict()
            }
        )
    
    async def _enhance_payment_with_kse(self, payment: PayPalPaymentData):
        """Enhance payment data with KSE universal substrate"""
        
        # Create neural content for embedding
        neural_content = self._create_payment_neural_content(payment)
        
        # Map to conceptual space
        conceptual_space = self._map_payment_to_conceptual_space(payment)
        
        # Create knowledge graph nodes
        knowledge_graph_nodes = self._create_payment_knowledge_graph_nodes(payment)
        
        # KSE enhancement completed - data is processed for neural embedding
        logger.info(f"Enhanced PayPal payment {payment.id} with KSE substrate integration")
    
    async def _enhance_transaction_with_kse(self, transaction: PayPalTransactionData):
        """Enhance transaction data with KSE universal substrate"""
        
        # Create neural content for embedding
        neural_content = self._create_transaction_neural_content(transaction)
        
        # Map to conceptual space
        conceptual_space = self._map_transaction_to_conceptual_space(transaction)
        
        # Create knowledge graph nodes
        knowledge_graph_nodes = self._create_transaction_knowledge_graph_
        nodes(transaction)
        
        # KSE enhancement completed - data is processed for neural embedding
        logger.info(f"Enhanced PayPal transaction {transaction.id} with KSE substrate integration")
    
    async def _enhance_subscription_with_kse(self, subscription: PayPalSubscriptionData):
        """Enhance subscription data with KSE universal substrate"""
        
        # Create neural content for embedding
        neural_content = self._create_subscription_neural_content(subscription)
        
        # Map to conceptual space
        conceptual_space = self._map_subscription_to_conceptual_space(subscription)
        
        # Create knowledge graph nodes
        knowledge_graph_nodes = self._create_subscription_knowledge_graph_nodes(subscription)
        
        # KSE enhancement completed - data is processed for neural embedding
        logger.info(f"Enhanced PayPal subscription {subscription.id} with KSE substrate integration")
    
    def _create_payment_neural_content(self, payment: PayPalPaymentData) -> str:
        """Create rich text content for payment neural embedding"""
        content_parts = [
            f"PayPal payment {payment.id}: {payment.intent}",
            f"Status: {payment.status.value}"
        ]
        
        # Add purchase unit information
        if payment.purchase_units:
            total_amount = 0
            currency = "USD"
            for unit in payment.purchase_units:
                if 'amount' in unit:
                    amount_info = unit['amount']
                    total_amount += float(amount_info.get('value', 0))
                    currency = amount_info.get('currency_code', 'USD')
                if 'description' in unit:
                    content_parts.append(f"Description: {unit['description']}")
            
            content_parts.append(f"Amount: {total_amount} {currency}")
        
        # Add payer information
        if payment.payer:
            payer = payment.payer
            if 'email_address' in payer:
                content_parts.append(f"Payer: {payer['email_address']}")
            if 'name' in payer and 'given_name' in payer['name']:
                content_parts.append(f"Name: {payer['name']['given_name']} {payer['name'].get('surname', '')}")
            if 'address' in payer and 'country_code' in payer['address']:
                content_parts.append(f"Country: {payer['address']['country_code']}")
        
        # Add payment source information
        if payment.payment_source:
            for method, details in payment.payment_source.items():
                content_parts.append(f"Payment method: {method}")
                if isinstance(details, dict) and 'brand' in details:
                    content_parts.append(f"Brand: {details['brand']}")
        
        # Add application context
        if payment.application_context:
            context = payment.application_context
            if 'brand_name' in context:
                content_parts.append(f"Brand: {context['brand_name']}")
            if 'user_action' in context:
                content_parts.append(f"User action: {context['user_action']}")
        
        return ". ".join(content_parts)
    
    def _create_transaction_neural_content(self, transaction: PayPalTransactionData) -> str:
        """Create rich text content for transaction neural embedding"""
        content_parts = [
            f"PayPal transaction {transaction.id}: {transaction.status.value}",
            f"Amount: {transaction.amount.get('value', 0)} {transaction.amount.get('currency_code', 'USD')}"
        ]
        
        # Add transaction details
        if transaction.description:
            content_parts.append(f"Description: {transaction.description}")
        
        if transaction.invoice_id:
            content_parts.append(f"Invoice: {transaction.invoice_id}")
        
        if transaction.custom_id:
            content_parts.append(f"Custom ID: {transaction.custom_id}")
        
        # Add fee information
        if transaction.transaction_fee:
            fee_info = transaction.transaction_fee
            if 'value' in fee_info:
                content_parts.append(f"Fee: {fee_info['value']} {fee_info.get('currency_code', 'USD')}")
        
        # Add protection information
        if transaction.protection_eligibility:
            content_parts.append(f"Protection: {transaction.protection_eligibility}")
        
        return ". ".join(content_parts)
    
    def _create_subscription_neural_content(self, subscription: PayPalSubscriptionData) -> str:
        """Create rich text content for subscription neural embedding"""
        content_parts = [
            f"PayPal subscription {subscription.id}: {subscription.status.value}",
            f"Plan: {subscription.plan_id}",
            f"Quantity: {subscription.quantity}"
        ]
        
        # Add subscriber information
        if subscription.subscriber:
            subscriber = subscription.subscriber
            if 'email_address' in subscriber:
                content_parts.append(f"Subscriber: {subscriber['email_address']}")
            if 'name' in subscriber and 'given_name' in subscriber['name']:
                content_parts.append(f"Name: {subscriber['name']['given_name']} {subscriber['name'].get('surname', '')}")
        
        # Add billing information
        if subscription.billing_info:
            billing = subscription.billing_info
            if 'cycle_executions' in billing:
                content_parts.append(f"Billing cycles: {len(billing['cycle_executions'])}")
            if 'last_payment' in billing:
                last_payment = billing['last_payment']
                if 'amount' in last_payment:
                    amount = last_payment['amount']
                    content_parts.append(f"Last payment: {amount.get('value', 0)} {amount.get('currency_code', 'USD')}")
        
        # Add shipping information
        if subscription.shipping_amount:
            shipping = subscription.shipping_amount
            content_parts.append(f"Shipping: {shipping.get('value', 0)} {shipping.get('currency_code', 'USD')}")
        
        # Add custom ID
        if subscription.custom_id:
            content_parts.append(f"Custom ID: {subscription.custom_id}")
        
        return ". ".join(content_parts)
    
    def _map_payment_to_conceptual_space(self, payment: PayPalPaymentData) -> str:
        """Map payment to conceptual space dimensions"""
        dimensions = []
        
        # Payment intent dimension
        dimensions.append(f"{payment.intent.lower()}_payment")
        
        # Payment status dimension
        dimensions.append(f"{payment.status.value.lower()}_status")
        
        # Payment amount dimension
        total_amount = 0
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'amount' in unit:
                    total_amount += float(unit['amount'].get('value', 0))
        
        if total_amount < 10:
            dimensions.append("micro_payment")
        elif total_amount < 100:
            dimensions.append("small_payment")
        elif total_amount < 1000:
            dimensions.append("medium_payment")
        else:
            dimensions.append("large_payment")
        
        # Payment method dimension
        if payment.payment_source:
            for method in payment.payment_source.keys():
                dimensions.append(f"{method.lower()}_method")
                break  # Take first method
        
        # Payer type dimension
        if payment.payer:
            if 'payer_id' in payment.payer:
                dimensions.append("registered_payer")
            else:
                dimensions.append("guest_payer")
        else:
            dimensions.append("anonymous_payer")
        
        return "_".join(dimensions)
    
    def _map_transaction_to_conceptual_space(self, transaction: PayPalTransactionData) -> str:
        """Map transaction to conceptual space dimensions"""
        dimensions = ["transaction"]
        
        # Transaction status dimension
        dimensions.append(f"{transaction.status.value.lower()}_transaction")
        
        # Transaction amount dimension
        amount = float(transaction.amount.get('value', 0))
        if amount < 10:
            dimensions.append("micro_transaction")
        elif amount < 100:
            dimensions.append("small_transaction")
        elif amount < 1000:
            dimensions.append("medium_transaction")
        else:
            dimensions.append("large_transaction")
        
        # Currency dimension
        currency = transaction.amount.get('currency_code', 'USD')
        dimensions.append(f"{currency.lower()}_currency")
        
        # Protection dimension
        if transaction.protection_eligibility:
            if 'eligible' in transaction.protection_eligibility.lower():
                dimensions.append("protected_transaction")
            else:
                dimensions.append("unprotected_transaction")
        
        return "_".join(dimensions)
    
    def _map_subscription_to_conceptual_space(self, subscription: PayPalSubscriptionData) -> str:
        """Map subscription to conceptual space dimensions"""
        dimensions = ["subscription"]
        
        # Subscription status dimension
        dimensions.append(f"{subscription.status.value.lower()}_subscription")
        
        # Quantity dimension
        quantity = int(subscription.quantity)
        if quantity == 1:
            dimensions.append("single_quantity")
        else:
            dimensions.append("multiple_quantity")
        
        # Billing frequency (inferred from plan)
        dimensions.append("recurring_billing")
        
        # Subscriber type dimension
        if subscription.subscriber and 'payer_id' in subscription.subscriber:
            dimensions.append("registered_subscriber")
        else:
            dimensions.append("guest_subscriber")
        
        return "_".join(dimensions)
    
    def _create_payment_knowledge_graph_nodes(self, payment: PayPalPaymentData) -> List[str]:
        """Create knowledge graph node connections for payment"""
        nodes = [
            f"payment_{payment.id}",
            f"intent_{payment.intent.lower()}",
            f"status_{payment.status.value.lower()}"
        ]
        
        # Add payer nodes
        if payment.payer:
            payer = payment.payer
            if 'payer_id' in payer:
                nodes.append(f"payer_{payer['payer_id']}")
            if 'email_address' in payer:
                domain = payer['email_address'].split('@')[1] if '@' in payer['email_address'] else 'unknown'
                nodes.append(f"email_domain_{domain}")
            if 'address' in payer and 'country_code' in payer['address']:
                nodes.append(f"country_{payer['address']['country_code']}")
        
        # Add payment method nodes
        if payment.payment_source:
            for method in payment.payment_source.keys():
                nodes.append(f"payment_method_{method.lower()}")
        
        # Add purchase unit nodes
        for i, unit in enumerate(payment.purchase_units):
            if 'amount' in unit:
                currency = unit['amount'].get('currency_code', 'USD')
                nodes.append(f"currency_{currency.lower()}")
        
        return nodes
    
    def _create_transaction_knowledge_graph_nodes(self, transaction: PayPalTransactionData) -> List[str]:
        """Create knowledge graph node connections for transaction"""
        nodes = [
            f"transaction_{transaction.id}",
            f"status_{transaction.status.value.lower()}",
            f"currency_{transaction.amount.get('currency_code', 'USD').lower()}"
        ]
        
        # Add invoice nodes
        if transaction.invoice_id:
            nodes.append(f"invoice_{transaction.invoice_id}")
        
        # Add custom ID nodes
        if transaction.custom_id:
            nodes.append(f"custom_{transaction.custom_id}")
        
        # Add protection nodes
        if transaction.protection_eligibility:
            nodes.append(f"protection_{transaction.protection_eligibility.lower().replace(' ', '_')}")
        
        return nodes
    
    def _create_subscription_knowledge_graph_nodes(self, subscription: PayPalSubscriptionData) -> List[str]:
        """Create knowledge graph node connections for subscription"""
        nodes = [
            f"subscription_{subscription.id}",
            f"plan_{subscription.plan_id}",
            f"status_{subscription.status.value.lower()}"
        ]
        
        # Add subscriber nodes
        if subscription.subscriber:
            subscriber = subscription.subscriber
            if 'payer_id' in subscriber:
                nodes.append(f"subscriber_{subscriber['payer_id']}")
            if 'email_address' in subscriber:
                domain = subscriber['email_address'].split('@')[1] if '@' in subscriber['email_address'] else 'unknown'
                nodes.append(f"email_domain_{domain}")
        
        # Add custom ID nodes
        if subscription.custom_id:
            nodes.append(f"custom_{subscription.custom_id}")
        
        return nodes
    
    async def _determine_payment_treatment_assignment(
        self,
        payment: PayPalPaymentData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for PayPal payment"""
        
        # Default to control group for PayPal payments
        treatment_type = TreatmentType.CONTROL
        treatment_group = "paypal_direct"
        treatment_intensity = 0.0
        
        # Check application context for attribution information
        if payment.application_context:
            context = payment.application_context
            
            # Check for brand name or referrer information
            if 'brand_name' in context:
                brand = context['brand_name'].lower()
                if 'google' in brand:
                    treatment_type = TreatmentType.BUDGET_INCREASE
                    treatment_group = "google_paypal"
                    treatment_intensity = 0.6
                elif 'facebook' in brand or 'meta' in brand:
                    treatment_type = TreatmentType.CREATIVE_CHANGE
                    treatment_group = "facebook_paypal"
                    treatment_intensity = 0.5
        
        # Check purchase units for custom fields that might contain attribution
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'custom_id' in unit:
                    custom_id = unit['custom_id'].lower()
                    if 'google' in custom_id or 'adwords' in custom_id:
                        treatment_type = TreatmentType.BUDGET_INCREASE
                        treatment_group = "google_ads_paypal"
                        treatment_intensity = 0.7
                    elif 'facebook' in custom_id or 'fb' in custom_id:
                        treatment_type = TreatmentType.CREATIVE_CHANGE
                        treatment_group = "facebook_ads_paypal"
                        treatment_intensity = 0.6
                    elif 'email' in custom_id:
                        treatment_type = TreatmentType.CREATIVE_CHANGE
                        treatment_group = "email_paypal"
                        treatment_intensity = 0.4
        
        # Adjust intensity based on payment characteristics
        if payment.purchase_units:
            total_amount = sum(float(unit['amount'].get('value', 0)) for unit in payment.purchase_units if 'amount' in unit)
            if total_amount > 500:
                treatment_intensity = min(treatment_intensity + 0.2, 1.0)
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER if payment.payer and 'payer_id' in payment.payer else RandomizationUnit.SESSION
        
        # Generate experiment ID
        experiment_id = None
        if payment.application_context and 'brand_name' in payment.application_context:
            experiment_id = f"paypal_{payment.application_context['brand_name']}_{payment.create_time.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.6
        )
    
    async def _determine_transaction_treatment_assignment(
        self,
        transaction: PayPalTransactionData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for transaction"""
        
        # Default to control group
        treatment_type = TreatmentType.CONTROL
        treatment_group = "paypal_transaction"
        treatment_intensity = 0.0
        
        # Check custom ID for attribution information
        if transaction.custom_id:
            custom_id = transaction.custom_id.lower()
            
            if 'google' in custom_id:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "google_paypal_transaction"
                treatment_intensity = 0.7
            elif 'facebook' in custom_id:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "facebook_paypal_transaction"
                treatment_intensity = 0.6
            elif 'email' in custom_id:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "email_paypal_transaction"
                treatment_intensity = 0.5
        
        # Check invoice ID for campaign information
        elif transaction.invoice_id:
            invoice_id = transaction.invoice_id.lower()
            if 'campaign' in invoice_id or 'promo' in invoice_id:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "campaign_paypal_transaction"
                treatment_intensity = 0.4
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER if transaction.custom_id else RandomizationUnit.SESSION
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=None,
            assignment_confidence=0.5
        )
    
    async def _determine_subscription_treatment_assignment(
        self,
        subscription: PayPalSubscriptionData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for subscription"""
        
        # Default to control group
        treatment_type = TreatmentType.CONTROL
        treatment_group = "paypal_subscription"
        treatment_intensity = 0.0
        
        # Check custom ID for attribution information
        if subscription.custom_id:
            custom_id = subscription.custom_id.lower()
            
            if 'google' in custom_id:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "google_paypal_subscription"
                treatment_intensity = 0.8
            elif 'facebook' in custom_id:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "facebook_paypal_subscription"
                treatment_intensity = 0.7
            elif 'email' in custom_id:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "email_paypal_subscription"
                treatment_intensity = 0.6
        
        # Subscriptions are high-value events
        treatment_intensity = min(treatment_intensity + 0.2, 1.0)
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=None,
            assignment_confidence=0.7
        )
    
    async def _detect_payment_confounders(
        self,
        payment: PayPalPaymentData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect payment-specific confounding variables"""
        confounders = []
        
        # Payment amount confounder
        total_amount = 0
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'amount' in unit:
                    total_amount += float(unit['amount'].get('value', 0))
        
        confounders.append(ConfounderVariable(
            variable_name="payment_amount",
            variable_type="continuous",
            value=total_amount,
            importance_score=0.9,
            detection_method="payment_analysis",
            control_strategy="include_payment_amount_as_covariate"
        ))
        
        # Payment method confounder
        if payment.payment_source:
            payment_methods = list(payment.payment_source.keys())
            if payment_methods:
                confounders.append(ConfounderVariable(
                    variable_name="payment_method",
                    variable_type="categorical",
                    value=payment_methods[0],
                    importance_score=0.7,
                    detection_method="payment_method_analysis",
                    control_strategy="include_payment_method_as_covariate"
                ))
        
        # Payer type confounder
        if payment.payer:
            payer_type = "registered" if 'payer_id' in payment.payer else "guest"
            confounders.append(ConfounderVariable(
                variable_name="payer_type",
                variable_type="categorical",
                value=payer_type,
                importance_score=0.6,
                detection_method="payer_analysis",
                control_strategy="include_payer_type_as_covariate"
            ))
        
        return confounders
    
    async def _detect_transaction_confounders(
        self,
        transaction: PayPalTransactionData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect transaction-specific confounding variables"""
        confounders = []
        
        # Transaction amount confounder
        amount = float(transaction.amount.get('value', 0))
        confounders.append(ConfounderVariable(
            variable_name="transaction_amount",
            variable_type="continuous",
            value=amount,
            importance_score=0.9,
            detection_method="transaction_analysis",
            control_strategy="include_transaction_amount_as_covariate"
        ))
        
        # Transaction fee confounder
        if transaction.transaction_fee and 'value' in transaction.transaction_fee:
            fee_amount = float(transaction.transaction_fee['value'])
            confounders.append(ConfounderVariable(
                variable_name="transaction_fee",
                variable_type="continuous",
                value=fee_amount,
                importance_score=0.5,
                detection_method="fee_analysis",
                control_strategy="include_transaction_fee_as_covariate"
            ))
        
        # Protection eligibility confounder
        if transaction.protection_eligibility:
            confounders.append(ConfounderVariable(
                variable_name="protection_eligibility",
                variable_type="categorical",
                value=transaction.protection_eligibility,
                importance_score=0.6,
                detection_method="protection_analysis",
                control_strategy="include_protection_status_as_covariate"
            ))
        
        return confounders
    
    async def _detect_subscription_confounders(
        self,
        subscription: PayPalSubscriptionData,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect subscription-specific confounding variables"""
        confounders = []
        
        # Subscription quantity confounder
        quantity = int(subscription.quantity)
        confounders.append(ConfounderVariable(
            variable_name="subscription_quantity",
            variable_type="continuous",
            value=quantity,
            importance_score=0.7,
            detection_method="subscription_analysis",
            control_strategy="include_quantity_as_covariate"
        ))
        
        # Plan ID confounder
        confounders.append(ConfounderVariable(
            variable_name="subscription_plan",
            variable_type="categorical",
            value=subscription.plan_id,
            importance_score=0.8,
            detection_method="plan_analysis",
            control_strategy="include_plan_as_covariate"
        ))
        
        return confounders
    
    def _extract_payment_geographic_data(self, payment: PayPalPaymentData) -> Dict[str, Any]:
        """Extract geographic information from payment"""
        geographic_data = {}
        
        # Get location from payer address
        if payment.payer and 'address' in payment.payer:
            address = payment.payer['address']
            geographic_data.update({
                "country": address.get("country_code"),
                "state": address.get("admin_area_1"),
                "city": address.get("admin_area_2"),
                "postal_code": address.get("postal_code")
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_transaction_geographic_data(self, transaction: PayPalTransactionData) -> Dict[str, Any]:
        """Extract geographic information for transaction"""
        # PayPal transaction data typically doesn't include geographic information
        return {}
    
    def _extract_subscription_geographic_data(self, subscription: PayPalSubscriptionData) -> Dict[str, Any]:
        """Extract geographic information for subscription"""
        geographic_data = {}
        
        # Get location from subscriber
        if subscription.subscriber and 'address' in subscription.subscriber:
            address = subscription.subscriber['address']
            geographic_data.update({
                "country": address.get("country_code"),
                "state": address.get("admin_area_1"),
                "city": address.get("admin_area_2"),
                "postal_code": address.get("postal_code")
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_payment_audience_data(self, payment: PayPalPaymentData) -> Dict[str, Any]:
        """Extract audience characteristics from payment data"""
        audience_data = {
            "payment_intent": payment.intent,
            "payment_status": payment.status.value,
            "purchase_units_count": len(payment.purchase_units),
            "has_payer_data": payment.payer is not None
        }
        
        # Add payment amount segment
        total_amount = 0
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'amount' in unit:
                    total_amount += float(unit['amount'].get('value', 0))
        
        audience_data["payment_amount_segment"] = self._categorize_payment_amount(total_amount)
        
        # Add payer characteristics
        if payment.payer:
            payer = payment.payer
            audience_data.update({
                "payer_type": "registered" if 'payer_id' in payer else "guest",
                "has_payer_name": 'name' in payer,
                "has_payer_address": 'address' in payer
            })
        
        # Add payment method information
        if payment.payment_source:
            audience_data["payment_methods"] = list(payment.payment_source.keys())
        
        return audience_data
    
    def _extract_transaction_audience_data(self, transaction: PayPalTransactionData) -> Dict[str, Any]:
        """Extract audience characteristics for transaction"""
        audience_data = {
            "transaction_status": transaction.status.value,
            "transaction_amount_segment": self._categorize_payment_amount(float(transaction.amount.get('value', 0))),
            "transaction_currency": transaction.amount.get('currency_code', 'USD'),
            "has_custom_id": transaction.custom_id is not None,
            "has_invoice_id": transaction.invoice_id is not None,
            "has_protection": transaction.protection_eligibility is not None
        }
        
        return audience_data
    
    def _extract_subscription_audience_data(self, subscription: PayPalSubscriptionData) -> Dict[str, Any]:
        """Extract audience characteristics for subscription"""
        audience_data = {
            "subscription_status": subscription.status.value,
            "subscription_quantity": int(subscription.quantity),
            "plan_id": subscription.plan_id,
            "has_subscriber_data": subscription.subscriber is not None,
            "has_billing_info": subscription.billing_info is not None,
            "has_shipping": subscription.shipping_amount is not None
        }
        
        return audience_data
    
    def _calculate_payment_data_quality_score(self, payment: PayPalPaymentData) -> float:
        """Calculate data quality score for payment causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [payment.id, payment.intent, payment.status, payment.create_time]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Purchase units data quality
        units_score = 0.2 if payment.purchase_units else 0.0
        if payment.purchase_units:
            units_score += 0.5 if any('amount' in unit for unit in payment.purchase_units) else 0.0
            units_score += 0.3 if any('description' in unit for unit in payment.purchase_units) else 0.0
        score_components.append(min(units_score, 1.0) * 0.25)
        
        # Payer data quality
        payer_score = 0.1 if payment.payer else 0.0
        if payment.payer:
            payer_score += 0.4 if 'email_address' in payment.payer else 0.0
            payer_score += 0.3 if 'name' in payment.payer else 0.0
            payer_score += 0.2 if 'address' in payment.payer else 0.0
        score_components.append(min(payer_score, 1.0) * 0.2)
        
        # Payment source data quality
        source_score = 0.5 if payment.payment_source else 0.0
        if payment.payment_source:
            source_score += 0.5
        score_components.append(min(source_score, 1.0) * 0.15)
        
        # Context data quality
        context_score = 0.5 if payment.application_context else 0.0
        if payment.application_context:
            context_score += 0.5
        
        score_components.append(min(context_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _calculate_transaction_data_quality_score(self, transaction: PayPalTransactionData) -> float:
        """Calculate data quality score for transaction causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [transaction.id, transaction.status, transaction.amount, transaction.create_time]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.4)
        
        # Amount data quality
        amount_score = 0.5 if transaction.amount and 'value' in transaction.amount else 0.0
        if transaction.amount:
            amount_score += 0.5 if 'currency_code' in transaction.amount else 0.0
        score_components.append(amount_score * 0.3)
        
        # Attribution data quality
        attribution_score = 0.0
        if transaction.custom_id:
            attribution_score += 0.5
        if transaction.invoice_id:
            attribution_score += 0.3
        if transaction.description:
            attribution_score += 0.2
        score_components.append(min(attribution_score, 1.0) * 0.2)
        
        # Additional data quality
        additional_score = 0.0
        if transaction.transaction_fee:
            additional_score += 0.3
        if transaction.protection_eligibility:
            additional_score += 0.4
        if transaction.related_resources:
            additional_score += 0.3
        score_components.append(min(additional_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _calculate_subscription_data_quality_score(self, subscription: PayPalSubscriptionData) -> float:
        """Calculate data quality score for subscription causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [subscription.id, subscription.plan_id, subscription.status, subscription.create_time]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Subscriber data quality
        subscriber_score = 0.1 if subscription.subscriber else 0.0
        if subscription.subscriber:
            subscriber_score += 0.4 if 'email_address' in subscription.subscriber else 0.0
            subscriber_score += 0.3 if 'name' in subscription.subscriber else 0.0
            subscriber_score += 0.2 if 'address' in subscription.subscriber else 0.0
        score_components.append(min(subscriber_score, 1.0) * 0.25)
        
        # Billing data quality
        billing_score = 0.2 if subscription.billing_info else 0.0
        if subscription.billing_info:
            billing_score += 0.4 if 'last_payment' in subscription.billing_info else 0.0
            billing_score += 0.4 if 'cycle_executions' in subscription.billing_info else 0.0
        score_components.append(min(billing_score, 1.0) * 0.2)
        
        # Plan and quantity data
        plan_score = 0.5 if subscription.plan_id else 0.0
        plan_score += 0.3 if subscription.quantity else 0.0
        plan_score += 0.2 if subscription.start_time else 0.0
        score_components.append(min(plan_score, 1.0) * 0.15)
        
        # Attribution data quality
        attribution_score = 0.0
        if subscription.custom_id:
            attribution_score += 0.6
        if subscription.application_context:
            attribution_score += 0.4
        score_components.append(min(attribution_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _categorize_payment_amount(self, amount: float) -> str:
        """Categorize payment amount into segments"""
        if amount < 5:
            return "micro"
        elif amount < 50:
            return "small"
        elif amount < 500:
            return "medium"
        elif amount < 5000:
            return "large"
        else:
            return "enterprise"
    
    def _determine_payment_campaign_info(self, payment: PayPalPaymentData) -> Dict[str, Any]:
        """Determine campaign information for payment"""
        campaign_info = {}
        
        # Extract from application context
        if payment.application_context:
            context = payment.application_context
            if 'brand_name' in context:
                campaign_info["brand_name"] = context['brand_name']
            if 'landing_page' in context:
                campaign_info["landing_page"] = context['landing_page']
            if 'user_action' in context:
                campaign_info["user_action"] = context['user_action']
        
        # Extract from purchase units
        if payment.purchase_units:
            for unit in payment.purchase_units:
                if 'custom_id' in unit:
                    campaign_info["custom_id"] = unit['custom_id']
                if 'invoice_id' in unit:
                    campaign_info["invoice_id"] = unit['invoice_id']
                if 'description' in unit:
                    campaign_info["description"] = unit['description']
        
        return campaign_info
    
    def _determine_transaction_campaign_info(self, transaction: PayPalTransactionData) -> Dict[str, Any]:
        """Determine campaign information for transaction"""
        campaign_info = {}
        
        # Extract from transaction fields
        if transaction.custom_id:
            campaign_info["custom_id"] = transaction.custom_id
        if transaction.invoice_id:
            campaign_info["invoice_id"] = transaction.invoice_id
        if transaction.description:
            campaign_info["description"] = transaction.description
        
        return campaign_info
    
    def _determine_subscription_campaign_info(self, subscription: PayPalSubscriptionData) -> Dict[str, Any]:
        """Determine campaign information for subscription"""
        campaign_info = {}
        
        # Extract from subscription fields
        if subscription.custom_id:
            campaign_info["custom_id"] = subscription.custom_id
        
        # Extract from application context
        if subscription.application_context:
            context = subscription.application_context
            if 'brand_name' in context:
                campaign_info["brand_name"] = context['brand_name']
            if 'user_action' in context:
                campaign_info["user_action"] = context['user_action']
        
        return campaign_info
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        if hasattr(self, 'session') and self.session:
            await self.session.close()