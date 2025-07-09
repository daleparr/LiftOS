"""
Test Suite for Tier 2 API Connectors (CRM and Payment Attribution)
LiftOS v1.3.0 - HubSpot, Salesforce, Stripe, PayPal Integration
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json
from typing import Dict, List, Any

# Import connectors
from connectors.hubspot_connector import HubSpotConnector
from connectors.salesforce_connector import SalesforceConnector
from connectors.stripe_connector import StripeConnector
from connectors.paypal_connector import PayPalConnector

# Import shared models
from shared.models.marketing import DataSource
from shared.models.causal_marketing import CausalMarketingData


class TestHubSpotConnector:
    """Test HubSpot CRM connector"""
    
    @pytest.fixture
    def hubspot_connector(self):
        return HubSpotConnector(api_key="test_api_key")
    
    @pytest.fixture
    def mock_hubspot_data(self):
        return {
            "deals": [
                {
                    "id": "123456789",
                    "properties": {
                        "dealname": "Test Deal",
                        "amount": "5000",
                        "dealstage": "closedwon",
                        "closedate": "2024-01-15T10:30:00Z",
                        "createdate": "2024-01-01T09:00:00Z",
                        "hs_deal_stage_probability": "1.0",
                        "hubspot_owner_id": "12345"
                    },
                    "associations": {
                        "companies": {"results": [{"id": "comp123"}]},
                        "contacts": {"results": [{"id": "cont456"}]}
                    }
                }
            ],
            "contacts": [
                {
                    "id": "cont456",
                    "properties": {
                        "firstname": "John",
                        "lastname": "Doe",
                        "email": "john.doe@example.com",
                        "lifecyclestage": "customer",
                        "hs_lead_status": "OPEN",
                        "hubspotscore": "85"
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, hubspot_connector, mock_hubspot_data):
        """Test HubSpot causal marketing data extraction"""
        with patch.object(hubspot_connector, '_fetch_deals', return_value=mock_hubspot_data["deals"]):
            with patch.object(hubspot_connector, '_fetch_contacts', return_value=mock_hubspot_data["contacts"]):
                with patch.object(hubspot_connector, '_fetch_companies', return_value=[]):
                    with patch.object(hubspot_connector, '_fetch_activities', return_value=[]):
                        
                        start_date = date(2024, 1, 1)
                        end_date = date(2024, 1, 31)
                        
                        causal_data = await hubspot_connector.extract_causal_marketing_data(
                            org_id="test_org",
                            start_date=start_date,
                            end_date=end_date,
                            historical_data=[]
                        )
                        
                        assert len(causal_data) > 0
                        assert all(isinstance(data, CausalMarketingData) for data in causal_data)
                        
                        # Check first record
                        first_record = causal_data[0]
                        assert first_record.data_source == DataSource.HUBSPOT
                        assert first_record.conversion_value > 0
                        assert first_record.data_quality_score > 0


class TestSalesforceConnector:
    """Test Salesforce CRM connector"""
    
    @pytest.fixture
    def salesforce_connector(self):
        return SalesforceConnector(
            username="test@example.com",
            password="password",
            security_token="token",
            client_id="client_id",
            client_secret="client_secret",
            is_sandbox=True
        )
    
    @pytest.fixture
    def mock_salesforce_data(self):
        return {
            "opportunities": [
                {
                    "Id": "0061234567890ABC",
                    "Name": "Test Opportunity",
                    "Amount": 10000.0,
                    "StageName": "Closed Won",
                    "CloseDate": "2024-01-15",
                    "CreatedDate": "2024-01-01T09:00:00.000+0000",
                    "Probability": 100,
                    "AccountId": "0011234567890DEF",
                    "OwnerId": "0051234567890GHI"
                }
            ],
            "leads": [
                {
                    "Id": "00Q1234567890JKL",
                    "FirstName": "Jane",
                    "LastName": "Smith",
                    "Email": "jane.smith@example.com",
                    "Status": "Qualified",
                    "LeadSource": "Web",
                    "Rating": "Hot",
                    "Score__c": 90
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, salesforce_connector, mock_salesforce_data):
        """Test Salesforce causal marketing data extraction"""
        with patch.object(salesforce_connector, '_execute_soql_query') as mock_query:
            mock_query.side_effect = [
                mock_salesforce_data["opportunities"],
                mock_salesforce_data["leads"],
                [],  # accounts
                [],  # contacts
                []   # activities
            ]
            
            start_date = date(2024, 1, 1)
            end_date = date(2024, 1, 31)
            
            causal_data = await salesforce_connector.extract_causal_marketing_data(
                org_id="test_org",
                start_date=start_date,
                end_date=end_date,
                historical_data=[]
            )
            
            assert len(causal_data) > 0
            assert all(isinstance(data, CausalMarketingData) for data in causal_data)
            
            # Check first record
            first_record = causal_data[0]
            assert first_record.data_source == DataSource.SALESFORCE
            assert first_record.conversion_value > 0


class TestStripeConnector:
    """Test Stripe payment connector"""
    
    @pytest.fixture
    def stripe_connector(self):
        return StripeConnector(api_key="sk_test_123456789")
    
    @pytest.fixture
    def mock_stripe_data(self):
        return {
            "payment_intents": [
                {
                    "id": "pi_1234567890",
                    "amount": 5000,
                    "currency": "usd",
                    "status": "succeeded",
                    "created": 1704067200,
                    "customer": "cus_1234567890",
                    "metadata": {
                        "utm_source": "google",
                        "utm_medium": "cpc",
                        "utm_campaign": "winter_sale"
                    }
                }
            ],
            "charges": [
                {
                    "id": "ch_1234567890",
                    "amount": 5000,
                    "currency": "usd",
                    "status": "succeeded",
                    "created": 1704067200,
                    "customer": "cus_1234567890",
                    "payment_intent": "pi_1234567890"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, stripe_connector, mock_stripe_data):
        """Test Stripe causal marketing data extraction"""
        with patch.object(stripe_connector, '_fetch_payment_intents', return_value=mock_stripe_data["payment_intents"]):
            with patch.object(stripe_connector, '_fetch_charges', return_value=mock_stripe_data["charges"]):
                with patch.object(stripe_connector, '_fetch_subscriptions', return_value=[]):
                    with patch.object(stripe_connector, '_fetch_customers', return_value=[]):
                        with patch.object(stripe_connector, '_fetch_invoices', return_value=[]):
                            
                            start_date = date(2024, 1, 1)
                            end_date = date(2024, 1, 31)
                            
                            causal_data = await stripe_connector.extract_causal_marketing_data(
                                org_id="test_org",
                                start_date=start_date,
                                end_date=end_date,
                                historical_data=[]
                            )
                            
                            assert len(causal_data) > 0
                            assert all(isinstance(data, CausalMarketingData) for data in causal_data)
                            
                            # Check first record
                            first_record = causal_data[0]
                            assert first_record.data_source == DataSource.STRIPE
                            assert first_record.conversion_value > 0


class TestPayPalConnector:
    """Test PayPal payment connector"""
    
    @pytest.fixture
    def paypal_connector(self):
        return PayPalConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            is_sandbox=True
        )
    
    @pytest.fixture
    def mock_paypal_data(self):
        return {
            "payments": [
                {
                    "id": "PAYID-1234567890",
                    "intent": "sale",
                    "state": "approved",
                    "create_time": "2024-01-15T10:30:00Z",
                    "transactions": [
                        {
                            "amount": {
                                "total": "50.00",
                                "currency": "USD"
                            },
                            "custom": "utm_source=facebook&utm_medium=cpc",
                            "invoice_number": "INV-001"
                        }
                    ],
                    "payer": {
                        "payer_info": {
                            "email": "customer@example.com",
                            "payer_id": "PAYER123456789"
                        }
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, paypal_connector, mock_paypal_data):
        """Test PayPal causal marketing data extraction"""
        with patch.object(paypal_connector, '_fetch_payments', return_value=mock_paypal_data["payments"]):
            with patch.object(paypal_connector, '_fetch_transactions', return_value=[]):
                with patch.object(paypal_connector, '_fetch_subscriptions', return_value=[]):
                    
                    start_date = date(2024, 1, 1)
                    end_date = date(2024, 1, 31)
                    
                    causal_data = await paypal_connector.extract_causal_marketing_data(
                        org_id="test_org",
                        start_date=start_date,
                        end_date=end_date,
                        historical_data=[]
                    )
                    
                    assert len(causal_data) > 0
                    assert all(isinstance(data, CausalMarketingData) for data in causal_data)
                    
                    # Check first record
                    first_record = causal_data[0]
                    assert first_record.data_source == DataSource.PAYPAL
                    assert first_record.conversion_value > 0


class TestTier2Integration:
    """Integration tests for Tier 2 connectors"""
    
    @pytest.mark.asyncio
    async def test_all_connectors_data_quality(self):
        """Test that all Tier 2 connectors produce high-quality causal data"""
        connectors = [
            HubSpotConnector(api_key="test"),
            SalesforceConnector("test", "test", "test", "test", "test"),
            StripeConnector(api_key="test"),
            PayPalConnector("test", "test")
        ]
        
        for connector in connectors:
            # Mock the data fetching methods
            with patch.object(connector, '_fetch_data', return_value=[]):
                causal_data = await connector.extract_causal_marketing_data(
                    org_id="test_org",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    historical_data=[]
                )
                
                # All connectors should return valid causal data structure
                assert isinstance(causal_data, list)
                for data in causal_data:
                    assert isinstance(data, CausalMarketingData)
                    assert data.data_quality_score >= 0.0
                    assert data.data_quality_score <= 1.0
    
    def test_connector_rate_limiting(self):
        """Test that connectors implement proper rate limiting"""
        # HubSpot: 600 requests per minute
        hubspot = HubSpotConnector(api_key="test")
        assert hasattr(hubspot, '_rate_limiter')
        
        # Salesforce: 4000 requests per hour  
        salesforce = SalesforceConnector("test", "test", "test", "test", "test")
        assert hasattr(salesforce, '_rate_limiter')
        
        # Stripe: 80 requests per second
        stripe = StripeConnector(api_key="test")
        assert hasattr(stripe, '_rate_limiter')
        
        # PayPal: 300 requests per minute
        paypal = PayPalConnector("test", "test")
        assert hasattr(paypal, '_rate_limiter')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])