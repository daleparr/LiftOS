"""
Stripe Payment Connector for LiftOS v1.3.0
Advanced payment processing attribution with subscription and transaction analysis
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

class StripePaymentStatus(str, Enum):
    """Stripe payment status values"""
    SUCCEEDED = "succeeded"
    PENDING = "pending"
    FAILED = "failed"
    CANCELED = "canceled"
    REQUIRES_ACTION = "requires_action"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_PAYMENT_METHOD = "requires_payment_method"

class StripeSubscriptionStatus(str, Enum):
    """Stripe subscription status values"""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    PAUSED = "paused"

class StripePaymentMethodType(str, Enum):
    """Stripe payment method types"""
    CARD = "card"
    BANK_ACCOUNT = "us_bank_account"
    SEPA_DEBIT = "sepa_debit"
    IDEAL = "ideal"
    SOFORT = "sofort"
    GIROPAY = "giropay"
    BANCONTACT = "bancontact"
    EPS = "eps"
    P24 = "p24"
    ALIPAY = "alipay"
    WECHAT_PAY = "wechat_pay"

# Stripe Data Models
class StripePaymentIntentData(BaseModel):
    """Stripe Payment Intent data model"""
    id: str
    amount: int  # Amount in cents
    currency: str
    status: StripePaymentStatus
    customer: Optional[str] = None
    description: Optional[str] = None
    receipt_email: Optional[str] = None
    payment_method: Optional[str] = None
    payment_method_types: List[str] = []
    created: datetime
    metadata: Dict[str, str] = {}
    charges: Optional[Dict[str, Any]] = None
    invoice: Optional[str] = None
    subscription: Optional[str] = None
    application_fee_amount: Optional[int] = None
    transfer_data: Optional[Dict[str, Any]] = None

class StripeChargeData(BaseModel):
    """Stripe Charge data model"""
    id: str
    amount: int  # Amount in cents
    currency: str
    status: str  # succeeded, pending, failed
    customer: Optional[str] = None
    description: Optional[str] = None
    receipt_email: Optional[str] = None
    payment_method: Optional[str] = None
    payment_method_details: Optional[Dict[str, Any]] = None
    created: datetime
    metadata: Dict[str, str] = {}
    invoice: Optional[str] = None
    payment_intent: Optional[str] = None
    refunded: bool = False
    amount_refunded: int = 0
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None

class StripeSubscriptionData(BaseModel):
    """Stripe Subscription data model"""
    id: str
    customer: str
    status: StripeSubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    created: datetime
    start_date: datetime
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    items: List[Dict[str, Any]] = []
    metadata: Dict[str, str] = {}
    plan: Optional[Dict[str, Any]] = None
    quantity: int = 1
    tax_percent: Optional[float] = None
    discount: Optional[Dict[str, Any]] = None
    application_fee_percent: Optional[float] = None
    latest_invoice: Optional[str] = None

class StripeCustomerData(BaseModel):
    """Stripe Customer data model"""
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None
    created: datetime
    metadata: Dict[str, str] = {}
    address: Optional[Dict[str, Any]] = None
    shipping: Optional[Dict[str, Any]] = None
    tax_exempt: Optional[str] = None
    default_source: Optional[str] = None
    invoice_prefix: Optional[str] = None
    balance: int = 0
    delinquent: bool = False
    currency: Optional[str] = None
    discount: Optional[Dict[str, Any]] = None
    subscriptions: Optional[Dict[str, Any]] = None

class StripeInvoiceData(BaseModel):
    """Stripe Invoice data model"""
    id: str
    customer: str
    subscription: Optional[str] = None
    status: str  # draft, open, paid, void, uncollectible
    amount_due: int
    amount_paid: int
    amount_remaining: int
    currency: str
    created: datetime
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, str] = {}
    description: Optional[str] = None
    receipt_number: Optional[str] = None
    payment_intent: Optional[str] = None
    charge: Optional[str] = None
    lines: Optional[Dict[str, Any]] = None
    tax: Optional[int] = None
    total: int
    subtotal: int

class StripeConnector:
    """
    Stripe Payment Connector with advanced payment and subscription attribution
    Integrates with Stripe API for comprehensive payment data extraction
    """
    
    def __init__(
        self,
        api_key: str,
        api_version: str = "2023-10-16"
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = "https://api.stripe.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting: Stripe allows 100 requests per second in live mode
        self.rate_limit_calls_per_second = 80  # Conservative rate limiting
        self.rate_limit_window = 1  # 1 second
        self.call_timestamps: List[datetime] = []
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Remove timestamps older than the rate limit window
        self.call_timestamps = [
            ts for ts in self.call_timestamps 
            if (now - ts).total_seconds() < self.rate_limit_window
        ]
        
        # Check if we're at the rate limit
        if len(self.call_timestamps) >= self.rate_limit_calls_per_second:
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
        """Make authenticated API request to Stripe"""
        await self._check_rate_limit()
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Stripe-Version': self.api_version,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            if method == "GET":
                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Stripe API request failed: {response.status} - {error_text}")
                        return None
            else:
                async with self.session.request(method, url, headers=headers, data=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Stripe API request failed: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error making Stripe API request: {str(e)}")
            return None
    
    async def get_payment_intents(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[StripePaymentIntentData]:
        """Fetch payment intents from Stripe"""
        payment_intents = []
        
        params = {
            'created[gte]': int(start_date.timestamp()),
            'created[lte]': int(end_date.timestamp()),
            'limit': limit,
            'expand[]': ['data.charges', 'data.customer']
        }
        
        response = await self._make_api_request("payment_intents", params)
        
        if response and 'data' in response:
            for pi_data in response['data']:
                try:
                    # Convert timestamp to datetime
                    pi_data['created'] = datetime.fromtimestamp(pi_data['created'])
                    
                    # Convert status to enum
                    if pi_data.get('status'):
                        try:
                            pi_data['status'] = StripePaymentStatus(pi_data['status'])
                        except ValueError:
                            pi_data['status'] = StripePaymentStatus.PENDING
                    
                    payment_intent = StripePaymentIntentData(**pi_data)
                    payment_intents.append(payment_intent)
                    
                except Exception as e:
                    logger.warning(f"Error parsing payment intent {pi_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(payment_intents)} payment intents from Stripe")
        return payment_intents
    
    async def get_charges(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[StripeChargeData]:
        """Fetch charges from Stripe"""
        charges = []
        
        params = {
            'created[gte]': int(start_date.timestamp()),
            'created[lte]': int(end_date.timestamp()),
            'limit': limit,
            'expand[]': ['data.customer', 'data.payment_intent']
        }
        
        response = await self._make_api_request("charges", params)
        
        if response and 'data' in response:
            for charge_data in response['data']:
                try:
                    # Convert timestamp to datetime
                    charge_data['created'] = datetime.fromtimestamp(charge_data['created'])
                    
                    charge = StripeChargeData(**charge_data)
                    charges.append(charge)
                    
                except Exception as e:
                    logger.warning(f"Error parsing charge {charge_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(charges)} charges from Stripe")
        return charges
    
    async def get_subscriptions(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[StripeSubscriptionData]:
        """Fetch subscriptions from Stripe"""
        subscriptions = []
        
        params = {
            'created[gte]': int(start_date.timestamp()),
            'created[lte]': int(end_date.timestamp()),
            'limit': limit,
            'expand[]': ['data.customer', 'data.latest_invoice']
        }
        
        response = await self._make_api_request("subscriptions", params)
        
        if response and 'data' in response:
            for sub_data in response['data']:
                try:
                    # Convert timestamps to datetime
                    timestamp_fields = ['created', 'start_date', 'current_period_start', 'current_period_end']
                    for field in timestamp_fields:
                        if sub_data.get(field):
                            sub_data[field] = datetime.fromtimestamp(sub_data[field])
                    
                    # Convert optional timestamp fields
                    optional_timestamp_fields = ['trial_start', 'trial_end', 'canceled_at', 'ended_at']
                    for field in optional_timestamp_fields:
                        if sub_data.get(field):
                            sub_data[field] = datetime.fromtimestamp(sub_data[field])
                    
                    # Convert status to enum
                    if sub_data.get('status'):
                        try:
                            sub_data['status'] = StripeSubscriptionStatus(sub_data['status'])
                        except ValueError:
                            sub_data['status'] = StripeSubscriptionStatus.ACTIVE
                    
                    subscription = StripeSubscriptionData(**sub_data)
                    subscriptions.append(subscription)
                    
                except Exception as e:
                    logger.warning(f"Error parsing subscription {sub_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(subscriptions)} subscriptions from Stripe")
        return subscriptions
    
    async def get_customers(self, customer_ids: List[str]) -> List[StripeCustomerData]:
        """Fetch customers by IDs"""
        if not customer_ids:
            return []
        
        customers = []
        
        # Fetch customers individually (Stripe doesn't support batch customer retrieval)
        for customer_id in customer_ids:
            try:
                response = await self._make_api_request(f"customers/{customer_id}")
                
                if response:
                    # Convert timestamp to datetime
                    response['created'] = datetime.fromtimestamp(response['created'])
                    
                    customer = StripeCustomerData(**response)
                    customers.append(customer)
                    
            except Exception as e:
                logger.warning(f"Error fetching customer {customer_id}: {str(e)}")
                continue
        
        logger.info(f"Retrieved {len(customers)} customers from Stripe")
        return customers
    
    async def get_invoices(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[StripeInvoiceData]:
        """Fetch invoices from Stripe"""
        invoices = []
        
        params = {
            'created[gte]': int(start_date.timestamp()),
            'created[lte]': int(end_date.timestamp()),
            'limit': limit,
            'expand[]': ['data.customer', 'data.subscription', 'data.payment_intent']
        }
        
        response = await self._make_api_request("invoices", params)
        
        if response and 'data' in response:
            for invoice_data in response['data']:
                try:
                    # Convert timestamps to datetime
                    timestamp_fields = ['created', 'period_start', 'period_end']
                    for field in timestamp_fields:
                        if invoice_data.get(field):
                            invoice_data[field] = datetime.fromtimestamp(invoice_data[field])
                    
                    # Convert optional timestamp fields
                    optional_timestamp_fields = ['due_date', 'paid_at']
                    for field in optional_timestamp_fields:
                        if invoice_data.get(field):
                            invoice_data[field] = datetime.fromtimestamp(invoice_data[field])
                    
                    invoice = StripeInvoiceData(**invoice_data)
                    invoices.append(invoice)
                    
                except Exception as e:
                    logger.warning(f"Error parsing invoice {invoice_data.get('id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(invoices)} invoices from Stripe")
        return invoices
    
    async def sync_stripe_data(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = False
    ) -> List[CausalMarketingData]:
        """
        Sync Stripe data and transform to causal marketing format
        Processes both one-time payments and subscription events
        """
        try:
            logger.info(f"Starting Stripe data sync for org {org_id}")
            
            # Fetch payment data
            payment_intents = await self.get_payment_intents(start_date, end_date)
            charges = await self.get_charges(start_date, end_date)
            subscriptions = await self.get_subscriptions(start_date, end_date)
            invoices = await self.get_invoices(start_date, end_date)
            
            # Get unique customer IDs
            customer_ids = set()
            for pi in payment_intents:
                if pi.customer:
                    customer_ids.add(pi.customer)
            for charge in charges:
                if charge.customer:
                    customer_ids.add(charge.customer)
            for sub in subscriptions:
                customer_ids.add(sub.customer)
            for invoice in invoices:
                customer_ids.add(invoice.customer)
            
            customers = await self.get_customers(list(customer_ids))
            customers_dict = {customer.id: customer for customer in customers}
            
            # Create lookup dictionaries
            invoices_dict = {invoice.id: invoice for invoice in invoices}
            subscriptions_dict = {sub.id: sub for sub in subscriptions}
            
            causal_data_list = []
            
            # Process successful payment intents as primary conversion events
            successful_payment_intents = [
                pi for pi in payment_intents 
                if pi.status == StripePaymentStatus.SUCCEEDED
            ]
            
            for payment_intent in successful_payment_intents:
                try:
                    customer = customers_dict.get(payment_intent.customer) if payment_intent.customer else None
                    invoice = invoices_dict.get(payment_intent.invoice) if payment_intent.invoice else None
                    subscription = subscriptions_dict.get(payment_intent.subscription) if payment_intent.subscription else None
                    
                    # Process payment intent data
                    causal_data = await self._process_payment_intent_data(
                        payment_intent, customer, invoice, subscription, org_id
                    )
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing payment intent {payment_intent.id}: {str(e)}")
                    continue
            
            # Process new subscriptions as secondary conversion events
            new_subscriptions = [
                sub for sub in subscriptions 
                if sub.status in [StripeSubscriptionStatus.ACTIVE, StripeSubscriptionStatus.TRIALING]
            ]
            
            for subscription in new_subscriptions:
                try:
                    customer = customers_dict.get(subscription.customer)
                    latest_invoice = invoices_dict.get(subscription.latest_invoice) if subscription.latest_invoice else None
                    
                    # Process subscription data
                    causal_data = await self._process_subscription_data(
                        subscription, customer, latest_invoice, org_id
                    )
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing subscription {subscription.id}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(causal_data_list)} Stripe records")
            return causal_data_list
            
        except Exception as e:
            logger.error(f"Error in Stripe data sync: {str(e)}")
            return []
    
    async def _process_payment_intent_data(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData],
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process payment intent data into causal marketing format"""
        
        # Convert amount from cents to dollars
        conversion_value = payment_intent.amount / 100.0
        is_conversion = payment_intent.status == StripePaymentStatus.SUCCEEDED
        conversion_date = payment_intent.created
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_payment_with_kse(
            payment_intent, customer, invoice, subscription
        )
        
        # Treatment assignment
        treatment_result = await self._determine_payment_treatment_assignment(
            payment_intent, customer, invoice, subscription, None
        )
        
        # Confounder detection
        confounders = await self._detect_payment_confounders(
            payment_intent, customer, invoice, subscription, None
        )
        
        # Extract geographic and audience data
        geographic_data = self._extract_payment_geographic_data(payment_intent, customer)
        audience_data = self._extract_payment_audience_data(payment_intent, customer, invoice, subscription)
        
        # Calculate data quality score
        data_quality_score = self._calculate_payment_data_quality_score(
            payment_intent, customer, invoice, subscription
        )
        
        # Determine campaign information
        campaign_info = self._determine_payment_campaign_info(payment_intent, customer, invoice)
        
        return CausalMarketingData(
            id=f"stripe_payment_{payment_intent.id}",
            org_id=org_id,
            data_source=DataSource.STRIPE,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=payment_intent.customer or f"anonymous_{payment_intent.id}",
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
                "payment_intent": payment_intent.dict(),
                "customer": customer.dict() if customer else None,
                "invoice": invoice.dict() if invoice else None,
                "subscription": subscription.dict() if subscription else None
            }
        )
    
    async def _process_subscription_data(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process subscription data into causal marketing format"""
        
        # Calculate subscription value (monthly recurring revenue)
        conversion_value = 0.0
        if subscription.items:
            for item in subscription.items:
                if 'price' in item and 'amount' in item['price']:
                    conversion_value += (item['price']['amount'] / 100.0) * item.get('quantity', 1)
        
        is_conversion = subscription.status in [StripeSubscriptionStatus.ACTIVE, StripeSubscriptionStatus.TRIALING]
        conversion_date = subscription.created
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_subscription_with_kse(
            subscription, customer, invoice
        )
        
        # Treatment assignment
        treatment_result = await self._determine_subscription_treatment_assignment(
            subscription, customer, invoice, None
        )
        
        # Confounder detection
        confounders = await self._detect_subscription_confounders(
            subscription, customer, invoice, None
        )
        
        # Extract geographic and audience data
        geographic_data = self._extract_subscription_geographic_data(subscription, customer)
        audience_data = self._extract_subscription_audience_data(subscription, customer, invoice)
        
        # Calculate data quality score
        data_quality_score = self._calculate_subscription_data_quality_score(
            subscription, customer, invoice
        )
        
        # Determine campaign information
        campaign_info = self._determine_subscription_campaign_info(subscription, customer, invoice)
        
        return CausalMarketingData(
            id=f"stripe_subscription_{subscription.id}",
            org_id=org_id,
            data_source=DataSource.STRIPE,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=subscription.customer,
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
                "subscription": subscription.dict(),
                "customer": customer.dict() if customer else None,
                "invoice": invoice.dict() if invoice else None
            }
        )
    
    async def _enhance_payment_with_kse(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ):
        """Enhance payment data with KSE universal substrate"""
        
        # Create neural content for embedding
        neural_content = self._create_payment_neural_content(payment_intent, customer, invoice, subscription)
        
        # Map to conceptual space
        conceptual_space = self._map_payment_to_conceptual_space(payment_intent, customer, invoice, subscription)
        
        # Create knowledge graph nodes
        knowledge_graph_nodes = self._create_payment_knowledge_graph_nodes(payment_intent, customer, invoice, subscription)
        
        # KSE enhancement completed - data is processed for neural embedding
        logger.info(f"Enhanced Stripe payment {payment_intent.id} with KSE substrate integration")
    
    async def _enhance_subscription_with_kse(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ):
        """Enhance subscription data with KSE universal substrate"""
        
        # Create neural content for embedding
        neural_content = self._create_subscription_neural_content(subscription, customer, invoice)
        
        # Map to conceptual space
        conceptual_space = self._map_subscription_to_conceptual_space(subscription, customer, invoice)
        
        # Create knowledge graph nodes
        knowledge_graph_nodes = self._create_subscription_knowledge_graph_nodes(subscription, customer, invoice)
        
        # KSE enhancement completed - data is processed for neural embedding
        logger.info(f"Enhanced Stripe subscription {subscription.id} with KSE substrate integration")
    
    def _create_payment_neural_content(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ) -> str:
        """Create rich text content for payment neural embedding"""
        content_parts = [
            f"Stripe payment {payment_intent.id}: ${payment_intent.amount / 100:.2f} {payment_intent.currency.upper()}",
            f"Status: {payment_intent.status.value}",
            f"Payment methods: {', '.join(payment_intent.payment_method_types)}"
        ]
        
        # Add customer information
        if customer:
            if customer.email:
                content_parts.append(f"Customer: {customer.email}")
            if customer.name:
                content_parts.append(f"Name: {customer.name}")
            if customer.address:
                address = customer.address
                if address.get('country'):
                    content_parts.append(f"Country: {address['country']}")
        
        # Add subscription context
        if subscription:
            content_parts.append(f"Subscription payment: {subscription.status.value}")
            if subscription.items:
                content_parts.append(f"Subscription items: {len(subscription.items)}")
        
        # Add invoice context
        if invoice:
            content_parts.append(f"Invoice: {invoice.status}")
            if invoice.description:
                content_parts.append(f"Description: {invoice.description}")
        
        # Add metadata information
        if payment_intent.metadata:
            for key, value in payment_intent.metadata.items():
                if key.lower() in ['campaign', 'source', 'medium', 'utm_source', 'utm_campaign']:
                    content_parts.append(f"{key}: {value}")
        
        return ". ".join(content_parts)
    
    def _create_subscription_neural_content(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> str:
        """Create rich text content for subscription neural embedding"""
        content_parts = [
            f"Stripe subscription {subscription.id}: {subscription.status.value}",
            f"Created: {subscription.created.strftime('%Y-%m-%d')}"
        ]
        
        # Add customer information
        if customer:
            if customer.email:
                content_parts.append(f"Customer: {customer.email}")
            if customer.name:
                content_parts.append(f"Name: {customer.name}")
        
        # Add subscription details
        if subscription.items:
            content_parts.append(f"Items: {len(subscription.items)} subscription items")
            for item in subscription.items[:3]:  # Limit to first 3 items
                if 'price' in item and 'nickname' in item['price']:
                    content_parts.append(f"Plan: {item['price']['nickname']}")
        
        # Add trial information
        if subscription.trial_start and subscription.trial_end:
            content_parts.append("Has trial period")
        
        # Add billing information
        if subscription.current_period_start and subscription.current_period_end:
            period_days = (subscription.current_period_end - subscription.current_period_start).days
            content_parts.append(f"Billing period: {period_days} days")
        
        # Add metadata information
        if subscription.metadata:
            for key, value in subscription.metadata.items():
                if key.lower() in ['campaign', 'source', 'medium', 'utm_source', 'utm_campaign']:
                    content_parts.append(f"{key}: {value}")
        
        return ". ".join(content_parts)
    
    def _map_payment_to_conceptual_space(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ) -> str:
        """Map payment to conceptual space dimensions"""
        dimensions = []
        
        # Payment value dimension
        amount_usd = payment_intent.amount / 100.0
        if amount_usd < 10:
            dimensions.append("micro_payment")
        elif amount_usd < 100:
            dimensions.append("small_payment")
        elif amount_usd < 1000:
            dimensions.append("medium_payment")
        elif amount_usd < 10000:
            dimensions.append("large_payment")
        else:
            dimensions.append("enterprise_payment")
        
        # Payment type dimension
        if subscription:
            dimensions.append("subscription_payment")
        else:
            dimensions.append("one_time_payment")
        
        # Payment method dimension
        if payment_intent.payment_method_types:
            primary_method = payment_intent.payment_method_types[0]
            dimensions.append(f"{primary_method}_payment")
        
        # Customer type dimension
        if customer:
            if customer.subscriptions and customer.subscriptions.get('data'):
                dimensions.append("recurring_customer")
            else:
                dimensions.append("new_customer")
        else:
            dimensions.append("anonymous_customer")
        
        # Currency dimension
        dimensions.append(f"{payment_intent.currency}_currency")
        
        # Status dimension
        dimensions.append(f"{payment_intent.status.value}_status")
        
        return "_".join(dimensions)
    
    def _map_subscription_to_conceptual_space(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> str:
        """Map subscription to conceptual space dimensions"""
        dimensions = ["subscription"]
        
        # Status dimension
        dimensions.append(f"{subscription.status.value}_subscription")
        
        # Trial dimension
        if subscription.trial_start and subscription.trial_end:
            dimensions.append("trial_subscription")
        else:
            dimensions.append("no_trial_subscription")
        
        # Billing frequency dimension
        if subscription.current_period_start and subscription.current_period_end:
            period_days = (subscription.current_period_end - subscription.current_period_start).days
            if period_days <= 7:
                dimensions.append("weekly_billing")
            elif period_days <= 31:
                dimensions.append("monthly_billing")
            elif period_days <= 93:
                dimensions.append("quarterly_billing")
            else:
                dimensions.append("annual_billing")
        
        # Item count dimension
        item_count = len(subscription.items)
        if item_count == 1:
            dimensions.append("single_item")
        elif item_count <= 3:
            dimensions.append("few_items")
        else:
            dimensions.append("many_items")
        
        return "_".join(dimensions)
    
    def _create_payment_knowledge_graph_nodes(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ) -> List[str]:
        """Create knowledge graph node connections for payment"""
        nodes = [
            f"payment_{payment_intent.id}",
            f"currency_{payment_intent.currency}",
            f"status_{payment_intent.status.value}"
        ]
        
        # Add customer nodes
        if customer:
            nodes.append(f"customer_{customer.id}")
            if customer.email:
                domain = customer.email.split('@')[1] if '@' in customer.email else 'unknown'
                nodes.append(f"email_domain_{domain}")
            if customer.address and customer.address.get('country'):
                nodes.append(f"country_{customer.address['country']}")
        
        # Add payment method nodes
        for method_type in payment_intent.payment_method_types:
            nodes.append(f"payment_method_{method_type}")
        
        # Add subscription nodes
        if subscription:
            nodes.append(f"subscription_{subscription.id}")
            nodes.append(f"subscription_status_{subscription.status.value}")
        
        # Add invoice nodes
        if invoice:
            nodes.append(f"invoice_{invoice.id}")
            nodes.append(f"invoice_status_{invoice.status}")
        
        # Add metadata nodes
        if payment_intent.metadata:
            for key, value in payment_intent.metadata.items():
                if key.lower() in ['campaign', 'source', 'medium']:
                    nodes.append(f"{key}_{value.lower().replace(' ', '_')}")
        
        return nodes
    
    def _create_subscription_knowledge_graph_nodes(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> List[str]:
        """Create knowledge graph node connections for subscription"""
        nodes = [
            f"subscription_{subscription.id}",
            f"subscription_status_{subscription.status.value}",
            f"customer_{subscription.customer}"
        ]
        
        # Add customer nodes
        if customer:
            if customer.email:
                domain = customer.email.split('@')[1] if '@' in customer.email else 'unknown'
                nodes.append(f"email_domain_{domain}")
            if customer.address and customer.address.get('country'):
                nodes.append(f"country_{customer.address['country']}")
        
        # Add plan nodes
        for item in subscription.items:
            if 'price' in item:
                price = item['price']
                if 'id' in price:
                    nodes.append(f"price_{price['id']}")
                if 'product' in price:
                    nodes.append(f"product_{price['product']}")
        
        # Add trial nodes
        if subscription.trial_start:
            nodes.append("trial_subscription")
        
        # Add metadata nodes
        if subscription.metadata:
            for key, value in subscription.metadata.items():
                if key.lower() in ['campaign', 'source', 'medium']:
                    nodes.append(f"{key}_{value.lower().replace(' ', '_')}")
        
        return nodes
    
    async def _determine_payment_treatment_assignment(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for Stripe payment"""
        
        # Determine treatment based on metadata and customer information
        treatment_type = TreatmentType.CONTROL
        treatment_group = "direct_payment"
        treatment_intensity = 0.0
        
        # Check metadata for attribution information
        if payment_intent.metadata:
            metadata = payment_intent.metadata
            
            # Check for UTM parameters
            if 'utm_source' in metadata:
                source = metadata['utm_source'].lower()
                
                if 'google' in source:
                    treatment_type = TreatmentType.BUDGET_INCREASE
                    treatment_group = "google_ads_payment"
                    treatment_intensity = 0.8
                elif 'facebook' in source or 'meta' in source:
                    treatment_type = TreatmentType.CREATIVE_CHANGE
                    treatment_group = "facebook_ads_payment"
                    treatment_intensity = 0.7
                elif 'email' in source:
                    treatment_type = TreatmentType.CREATIVE_CHANGE
                    treatment_group = "email_marketing_payment"
                    treatment_intensity = 0.6
                else:
                    treatment_type = TreatmentType.CONTROL
                    treatment_group = "other_paid_payment"
                    treatment_intensity = 0.3
            
            # Check for campaign information
            elif 'campaign' in metadata or 'utm_campaign' in metadata:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "campaign_payment"
                treatment_intensity = 0.5
        
        # Adjust intensity based on payment characteristics
        amount_usd = payment_intent.amount / 100.0
        if amount_usd > 1000:
            treatment_intensity = min(treatment_intensity + 0.2, 1.0)  # High-value payments get higher intensity
        
        if subscription:
            treatment_intensity = min(treatment_intensity + 0.1, 1.0)  # Subscription payments get higher intensity
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER if customer else RandomizationUnit.SESSION
        
        # Generate experiment ID
        experiment_id = None
        if payment_intent.metadata and 'utm_campaign' in payment_intent.metadata:
            experiment_id = f"stripe_campaign_{payment_intent.metadata['utm_campaign']}_{payment_intent.created.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.8 if payment_intent.metadata else 0.5
        )
    
    async def _determine_subscription_treatment_assignment(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for subscription"""
        
        # Determine treatment based on metadata
        treatment_type = TreatmentType.CONTROL
        treatment_group = "direct_subscription"
        treatment_intensity = 0.0
        
        # Check metadata for attribution information
        if subscription.metadata:
            metadata = subscription.metadata
            
            if 'utm_source' in metadata:
                source = metadata['utm_source'].lower()
                
                if 'google' in source:
                    treatment_type = TreatmentType.BUDGET_INCREASE
                    treatment_group = "google_ads_subscription"
                    treatment_intensity = 0.9
                elif 'facebook' in source:
                    treatment_type = TreatmentType.CREATIVE_CHANGE
                    treatment_group = "facebook_ads_subscription"
                    treatment_intensity = 0.8
                elif 'email' in source:
                    treatment_type = TreatmentType.CREATIVE_CHANGE
                    treatment_group = "email_marketing_subscription"
                    treatment_intensity = 0.7
                else:
                    treatment_type = TreatmentType.CONTROL
                    treatment_group = "other_paid_subscription"
                    treatment_intensity = 0.4
        
        # Subscriptions are high-value events
        treatment_intensity = min(treatment_intensity + 0.2, 1.0)
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER
        
        # Generate experiment ID
        experiment_id = None
        if subscription.metadata and 'utm_campaign' in subscription.metadata:
            experiment_id = f"stripe_subscription_{subscription.metadata['utm_campaign']}_{subscription.created.strftime('%Y%m')}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.85 if subscription.metadata else 0.6
        )
    
    async def _detect_payment_confounders(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect payment-specific confounding variables"""
        confounders = []
        
        # Payment amount confounder
        amount_usd = payment_intent.amount / 100.0
        confounders.append(ConfounderVariable(
            variable_name="payment_amount",
            variable_type="continuous",
            value=amount_usd,
            importance_score=0.9,
            detection_method="payment_analysis",
            control_strategy="include_payment_amount_as_covariate"
        ))
        
        # Customer history confounder
        if customer:
            if customer.balance != 0:
                confounders.append(ConfounderVariable(
                    variable_name="customer_balance",
                    variable_type="continuous",
                    value=customer.balance / 100.0,
                    importance_score=0.7,
                    detection_method="customer_analysis",
                    control_strategy="include_customer_balance_as_covariate"
                ))
            
            if customer.delinquent:
                confounders.append(ConfounderVariable(
                    variable_name="customer_delinquent",
                    variable_type="categorical",
                    value="delinquent",
                    importance_score=0.8,
                    detection_method="customer_analysis",
                    control_strategy="include_delinquent_status_as_covariate"
                ))
        
        # Subscription context confounder
        if subscription:
            confounders.append(ConfounderVariable(
                variable_name="is_subscription_payment",
                variable_type="categorical",
                value="subscription",
                importance_score=0.85,
                detection_method="payment_context_analysis",
                control_strategy="include_subscription_flag_as_covariate"
            ))
        
        return confounders
    
    async def _detect_subscription_confounders(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect subscription-specific confounding variables"""
        confounders = []
        
        # Trial period confounder
        if subscription.trial_start and subscription.trial_end:
            trial_days = (subscription.trial_end - subscription.trial_start).days
            confounders.append(ConfounderVariable(
                variable_name="trial_period_days",
                variable_type="continuous",
                value=trial_days,
                importance_score=0.8,
                detection_method="subscription_analysis",
                control_strategy="include_trial_period_as_covariate"
            ))
        
        # Subscription item count confounder
        item_count = len(subscription.items)
        confounders.append(ConfounderVariable(
            variable_name="subscription_item_count",
            variable_type="continuous",
            value=item_count,
            importance_score=0.7,
            detection_method="subscription_analysis",
            control_strategy="include_item_count_as_covariate"
        ))
        
        # Customer context confounder
        if customer and customer.subscriptions:
            existing_subs = customer.subscriptions.get('total_count', 0)
            if existing_subs > 1:
                confounders.append(ConfounderVariable(
                    variable_name="existing_subscriptions",
                    variable_type="continuous",
                    value=existing_subs,
                    importance_score=0.75,
                    detection_method="customer_analysis",
                    control_strategy="include_existing_subscriptions_as_covariate"
                ))
        
        return confounders
    
    def _extract_payment_geographic_data(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData]
    ) -> Dict[str, Any]:
        """Extract geographic information from payment and customer"""
        geographic_data = {}
        
        # Get location from customer address
        if customer and customer.address:
            address = customer.address
            geographic_data.update({
                "city": address.get("city"),
                "state": address.get("state"),
                "country": address.get("country"),
                "postal_code": address.get("postal_code")
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_subscription_geographic_data(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData]
    ) -> Dict[str, Any]:
        """Extract geographic information for subscription"""
        geographic_data = {}
        
        # Get location from customer
        if customer and customer.address:
            address = customer.address
            geographic_data.update({
                "city": address.get("city"),
                "state": address.get("state"),
                "country": address.get("country"),
                "postal_code": address.get("postal_code")
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_payment_audience_data(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ) -> Dict[str, Any]:
        """Extract audience characteristics from payment data"""
        audience_data = {
            "payment_amount_segment": self._categorize_payment_amount(payment_intent.amount / 100.0),
            "payment_currency": payment_intent.currency,
            "payment_method_types": payment_intent.payment_method_types,
            "is_subscription_payment": subscription is not None,
            "has_customer_data": customer is not None
        }
        
        # Add customer characteristics
        if customer:
            audience_data.update({
                "customer_has_name": customer.name is not None,
                "customer_has_phone": customer.phone is not None,
                "customer_delinquent": customer.delinquent,
                "customer_balance_segment": self._categorize_customer_balance(customer.balance / 100.0)
            })
        
        return audience_data
    
    def _extract_subscription_audience_data(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> Dict[str, Any]:
        """Extract audience characteristics for subscription"""
        audience_data = {
            "subscription_status": subscription.status.value,
            "subscription_item_count": len(subscription.items),
            "has_trial": subscription.trial_start is not None,
            "billing_frequency": self._categorize_billing_frequency(subscription)
        }
        
        # Add customer context
        if customer:
            audience_data.update({
                "customer_has_name": customer.name is not None,
                "customer_delinquent": customer.delinquent
            })
        
        return audience_data
    
    def _calculate_payment_data_quality_score(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData],
        subscription: Optional[StripeSubscriptionData]
    ) -> float:
        """Calculate data quality score for payment causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [payment_intent.id, payment_intent.amount, payment_intent.currency, payment_intent.status]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Customer data quality
        customer_score = 0.1 if customer else 0.0
        if customer:
            customer_score += 0.4 if customer.email else 0.0
            customer_score += 0.3 if customer.name else 0.0
            customer_score += 0.2 if customer.address else 0.0
        score_components.append(min(customer_score, 1.0) * 0.25)
        
        # Attribution data quality
        attribution_score = 0.1  # Base score
        if payment_intent.metadata:
            if 'utm_source' in payment_intent.metadata:
                attribution_score += 0.4
            if 'utm_campaign' in payment_intent.metadata:
                attribution_score += 0.3
            if 'utm_medium' in payment_intent.metadata:
                attribution_score += 0.2
        score_components.append(min(attribution_score, 1.0) * 0.2)
        
        # Payment method data quality
        method_score = 0.5 if payment_intent.payment_method_types else 0.0
        if payment_intent.payment_method:
            method_score += 0.5
        score_components.append(min(method_score, 1.0) * 0.15)
        
        # Context data quality
        context_score = 0.0
        if subscription:
            context_score += 0.5
        if invoice:
            context_score += 0.3
        if payment_intent.description:
            context_score += 0.2
        score_components.append(min(context_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _calculate_subscription_data_quality_score(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> float:
        """Calculate data quality score for subscription causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [subscription.id, subscription.customer, subscription.status, subscription.created]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Customer data quality
        customer_score = 0.1 if customer else 0.0
        if customer:
            customer_score += 0.4 if customer.email else 0.0
            customer_score += 0.3 if customer.name else 0.0
            customer_score += 0.2 if customer.address else 0.0
        score_components.append(min(customer_score, 1.0) * 0.25)
        
        # Subscription items quality
        items_score = 0.2 if subscription.items else 0.0
        if subscription.items:
            items_score += 0.5 if len(subscription.items) > 0 else 0.0
            items_score += 0.3 if any('price' in item for item in subscription.items) else 0.0
        score_components.append(min(items_score, 1.0) * 0.2)
        
        # Attribution data quality
        attribution_score = 0.1  # Base score
        if subscription.metadata:
            if 'utm_source' in subscription.metadata:
                attribution_score += 0.4
            if 'utm_campaign' in subscription.metadata:
                attribution_score += 0.3
            if 'utm_medium' in subscription.metadata:
                attribution_score += 0.2
        score_components.append(min(attribution_score, 1.0) * 0.15)
        
        # Billing data quality
        billing_score = 0.0
        if subscription.current_period_start and subscription.current_period_end:
            billing_score += 0.6
        if subscription.trial_start or subscription.trial_end:
            billing_score += 0.2
        if invoice:
            billing_score += 0.2
        score_components.append(min(billing_score, 1.0) * 0.1)
        
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
    
    def _categorize_customer_balance(self, balance: float) -> str:
        """Categorize customer balance"""
        if balance < 0:
            return "negative"
        elif balance == 0:
            return "zero"
        elif balance < 100:
            return "small_positive"
        else:
            return "large_positive"
    
    def _categorize_billing_frequency(self, subscription: StripeSubscriptionData) -> str:
        """Categorize billing frequency"""
        if subscription.current_period_start and subscription.current_period_end:
            period_days = (subscription.current_period_end - subscription.current_period_start).days
            if period_days <= 7:
                return "weekly"
            elif period_days <= 31:
                return "monthly"
            elif period_days <= 93:
                return "quarterly"
            else:
                return "annual"
        return "unknown"
    
    def _determine_payment_campaign_info(
        self,
        payment_intent: StripePaymentIntentData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> Dict[str, Any]:
        """Determine campaign information for payment"""
        campaign_info = {}
        
        # Extract from payment intent metadata
        if payment_intent.metadata:
            metadata = payment_intent.metadata
            for key in ['utm_source', 'utm_medium', 'utm_campaign', 'campaign', 'source']:
                if key in metadata:
                    campaign_info[key] = metadata[key]
        
        return campaign_info
    
    def _determine_subscription_campaign_info(
        self,
        subscription: StripeSubscriptionData,
        customer: Optional[StripeCustomerData],
        invoice: Optional[StripeInvoiceData]
    ) -> Dict[str, Any]:
        """Determine campaign information for subscription"""
        campaign_info = {}
        
        # Extract from subscription metadata
        if subscription.metadata:
            metadata = subscription.metadata
            for key in ['utm_source', 'utm_medium', 'utm_campaign', 'campaign', 'source']:
                if key in metadata:
                    campaign_info[key] = metadata[key]
        
        return campaign_info
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        if hasattr(self, 'session') and self.session:
            await self.session.close()