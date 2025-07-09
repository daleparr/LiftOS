"""
Test Suite for Tier 1 API Connectors with KSE Integration
Tests Shopify, WooCommerce, and Amazon Seller Central connectors
"""
import asyncio
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import connectors
from connectors.shopify_connector import ShopifyConnector
from connectors.woocommerce_connector import WooCommerceConnector
from connectors.amazon_connector import AmazonConnector

# Import shared models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.causal_marketing import (
    CausalMarketingData, TreatmentType, RandomizationUnit, 
    TreatmentAssignmentResult, ConfounderVariable
)


class TestShopifyConnector:
    """Test Shopify connector with KSE integration"""
    
    @pytest.fixture
    def shopify_connector(self):
        """Create Shopify connector instance"""
        return ShopifyConnector(
            shop_domain="test-shop.myshopify.com",
            access_token="test_access_token"
        )
    
    @pytest.fixture
    def mock_shopify_order(self):
        """Mock Shopify order data"""
        return {
            "id": 12345,
            "order_number": "1001",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "total_price": "99.99",
            "subtotal_price": "89.99",
            "total_tax": "10.00",
            "currency": "USD",
            "financial_status": "paid",
            "fulfillment_status": "fulfilled",
            "customer": {
                "id": 67890,
                "email": "customer@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "orders_count": 3,
                "total_spent": "299.97"
            },
            "line_items": [
                {
                    "id": 11111,
                    "product_id": 22222,
                    "variant_id": 33333,
                    "title": "Test Product",
                    "quantity": 1,
                    "price": "89.99",
                    "sku": "TEST-SKU-001"
                }
            ],
            "shipping_address": {
                "city": "New York",
                "province": "NY",
                "country": "United States",
                "zip": "10001"
            },
            "note": "utm_source=facebook&utm_medium=cpc&utm_campaign=summer_sale",
            "referring_site": "https://facebook.com/",
            "source_name": "facebook"
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, shopify_connector, mock_shopify_order):
        """Test extraction of causal marketing data from Shopify"""
        with patch.object(shopify_connector.client, 'get') as mock_get:
            # Mock API responses
            mock_get.side_effect = [
                # Orders response
                MagicMock(json=lambda: {"orders": [mock_shopify_order]}),
                # Products response
                MagicMock(json=lambda: {"products": [
                    {
                        "id": 22222,
                        "title": "Test Product",
                        "product_type": "Electronics",
                        "vendor": "Test Vendor",
                        "created_at": "2024-01-01T00:00:00Z"
                    }
                ]}),
                # Customers response
                MagicMock(json=lambda: {"customers": [
                    {
                        "id": 67890,
                        "email": "customer@example.com",
                        "created_at": "2023-12-01T00:00:00Z",
                        "orders_count": 3,
                        "total_spent": "299.97"
                    }
                ]})
            ]
            
            # Mock KSE client
            with patch('connectors.shopify_connector.KSEClient') as mock_kse:
                mock_kse_instance = AsyncMock()
                mock_kse.return_value = mock_kse_instance
                mock_kse_instance.store_causal_memory.return_value = "memory_123"
                
                # Execute extraction
                causal_data_list = await shopify_connector.extract_causal_marketing_data(
                    org_id="test_org",
                    start_date=date(2024, 1, 15),
                    end_date=date(2024, 1, 15),
                    historical_data=[]
                )
                
                # Assertions
                assert len(causal_data_list) == 1
                causal_data = causal_data_list[0]
                
                assert isinstance(causal_data, CausalMarketingData)
                assert causal_data.record_id == "12345"
                assert causal_data.treatment_type == TreatmentType.BUDGET_INCREASE
                assert causal_data.randomization_unit == RandomizationUnit.CUSTOMER
                assert causal_data.outcome_value == 99.99
                assert "utm_source=facebook" in causal_data.causal_metadata["utm_parameters"]
                assert causal_data.data_quality_score > 0.8
    
    @pytest.mark.asyncio
    async def test_treatment_assignment(self, shopify_connector, mock_shopify_order):
        """Test treatment assignment logic"""
        from connectors.shopify_connector import ShopifyOrderData
        
        order_data = ShopifyOrderData(**mock_shopify_order)
        
        treatment_result = await shopify_connector._determine_treatment_assignment(
            order_data, [], None
        )
        
        assert isinstance(treatment_result, TreatmentAssignmentResult)
        assert treatment_result.treatment_type == TreatmentType.BUDGET_INCREASE
        assert treatment_result.treatment_group == "facebook_ads"
        assert treatment_result.assignment_confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_confounder_detection(self, shopify_connector, mock_shopify_order):
        """Test confounder detection"""
        from connectors.shopify_connector import ShopifyOrderData, ShopifyCustomerData
        
        order_data = ShopifyOrderData(**mock_shopify_order)
        customer_data = ShopifyCustomerData(**mock_shopify_order["customer"])
        
        confounders = await shopify_connector._detect_shopify_confounders(
            order_data, customer_data, []
        )
        
        assert len(confounders) > 0
        assert any(c.variable_name == "customer_ltv" for c in confounders)
        assert any(c.variable_name == "repeat_customer" for c in confounders)


class TestWooCommerceConnector:
    """Test WooCommerce connector with KSE integration"""
    
    @pytest.fixture
    def woocommerce_connector(self):
        """Create WooCommerce connector instance"""
        return WooCommerceConnector(
            site_url="https://test-site.com",
            consumer_key="test_consumer_key",
            consumer_secret="test_consumer_secret"
        )
    
    @pytest.fixture
    def mock_woocommerce_order(self):
        """Mock WooCommerce order data"""
        return {
            "id": 54321,
            "number": "2001",
            "date_created": "2024-01-15T10:30:00",
            "date_modified": "2024-01-15T10:30:00",
            "status": "completed",
            "currency": "USD",
            "total": "149.99",
            "subtotal": "139.99",
            "total_tax": "10.00",
            "customer_id": 98765,
            "billing": {
                "first_name": "Jane",
                "last_name": "Smith",
                "email": "jane@example.com",
                "city": "Los Angeles",
                "state": "CA",
                "country": "US",
                "postcode": "90210"
            },
            "line_items": [
                {
                    "id": 44444,
                    "product_id": 55555,
                    "name": "Premium Widget",
                    "quantity": 2,
                    "price": 69.99,
                    "sku": "WIDGET-PREM-001"
                }
            ],
            "meta_data": [
                {"key": "_utm_source", "value": "google"},
                {"key": "_utm_medium", "value": "cpc"},
                {"key": "_utm_campaign", "value": "winter_promo"},
                {"key": "_referring_post", "value": "123"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, woocommerce_connector, mock_woocommerce_order):
        """Test extraction of causal marketing data from WooCommerce"""
        with patch.object(woocommerce_connector.client, 'get') as mock_get:
            # Mock API responses
            mock_get.side_effect = [
                # Orders response
                MagicMock(json=lambda: [mock_woocommerce_order]),
                # Products response
                MagicMock(json=lambda: [
                    {
                        "id": 55555,
                        "name": "Premium Widget",
                        "type": "simple",
                        "categories": [{"name": "Widgets"}],
                        "date_created": "2024-01-01T00:00:00"
                    }
                ]),
                # WordPress posts response
                MagicMock(json=lambda: [
                    {
                        "id": 123,
                        "title": {"rendered": "How to Use Widgets"},
                        "date": "2024-01-10T00:00:00",
                        "categories": [1],
                        "meta": {"views": 1500, "shares": 25}
                    }
                ])
            ]
            
            # Mock KSE client
            with patch('connectors.woocommerce_connector.KSEClient') as mock_kse:
                mock_kse_instance = AsyncMock()
                mock_kse.return_value = mock_kse_instance
                mock_kse_instance.store_causal_memory.return_value = "memory_456"
                
                # Execute extraction
                causal_data_list = await woocommerce_connector.extract_causal_marketing_data(
                    org_id="test_org",
                    start_date=date(2024, 1, 15),
                    end_date=date(2024, 1, 15),
                    historical_data=[]
                )
                
                # Assertions
                assert len(causal_data_list) == 1
                causal_data = causal_data_list[0]
                
                assert isinstance(causal_data, CausalMarketingData)
                assert causal_data.record_id == "54321"
                assert causal_data.treatment_type == TreatmentType.CONTENT_MARKETING
                assert causal_data.outcome_value == 149.99
                assert "content_influenced_purchase" in causal_data.causal_metadata
    
    @pytest.mark.asyncio
    async def test_content_attribution(self, woocommerce_connector):
        """Test content marketing attribution"""
        from connectors.woocommerce_connector import WooCommerceOrderData, WordPressPostData
        
        order_data = WooCommerceOrderData(
            id=54321,
            number="2001",
            date_created=datetime(2024, 1, 15, 10, 30),
            status="completed",
            currency="USD",
            total=149.99,
            customer_id=98765,
            meta_data=[{"key": "_referring_post", "value": "123"}]
        )
        
        post_data = WordPressPostData(
            id=123,
            title="How to Use Widgets",
            date=datetime(2024, 1, 10),
            categories=[1],
            views=1500,
            shares=25
        )
        
        attribution_score = woocommerce_connector._calculate_content_attribution_score(
            order_data, post_data
        )
        
        assert 0.0 <= attribution_score <= 1.0
        assert attribution_score > 0.5  # Should be high due to direct reference


class TestAmazonConnector:
    """Test Amazon Seller Central connector with KSE integration"""
    
    @pytest.fixture
    def amazon_connector(self):
        """Create Amazon connector instance"""
        return AmazonConnector(
            marketplace_id="ATVPDKIKX0DER",
            seller_id="test_seller_id",
            aws_access_key="test_access_key",
            aws_secret_key="test_secret_key",
            role_arn="arn:aws:iam::123456789:role/test-role",
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token"
        )
    
    @pytest.fixture
    def mock_amazon_sale(self):
        """Mock Amazon sales data"""
        return {
            "order_id": "123-4567890-1234567",
            "purchase_date": "2024-01-15T10:30:00Z",
            "asin": "B08N5WRWNW",
            "sku": "AMZN-SKU-001",
            "product_name": "Amazing Product",
            "quantity": 1,
            "item_price": 29.99,
            "currency": "USD",
            "marketplace_id": "ATVPDKIKX0DER",
            "marketplace_name": "Amazon.com",
            "fulfillment_channel": "AFN",
            "ship_city": "Seattle",
            "ship_state": "WA",
            "ship_country": "US",
            "is_business_order": False,
            "item_promotion_discount": 5.00,
            "buy_box_percentage": 85.5,
            "category_rank": 1250,
            "competitor_rank": 15
        }
    
    @pytest.fixture
    def mock_amazon_ad_data(self):
        """Mock Amazon advertising data"""
        return {
            "campaign_id": "12345678901234567890",
            "campaign_name": "Product Launch Campaign",
            "campaign_type": "sponsoredProducts",
            "ad_group_id": "98765432109876543210",
            "ad_group_name": "Main Ad Group",
            "targeting_type": "auto",
            "match_type": "broad",
            "keyword": "amazing product",
            "impressions": 10000,
            "clicks": 250,
            "cost": 75.50,
            "sales": 899.75,
            "orders": 30,
            "acos": 8.4,
            "roas": 11.92,
            "date": "2024-01-15"
        }
    
    @pytest.mark.asyncio
    async def test_extract_causal_marketing_data(self, amazon_connector, mock_amazon_sale, mock_amazon_ad_data):
        """Test extraction of causal marketing data from Amazon"""
        with patch.object(amazon_connector, '_get_access_token') as mock_token:
            mock_token.return_value = "test_access_token"
            
            with patch.object(amazon_connector.client, 'get') as mock_get:
                # Mock API responses
                mock_get.side_effect = [
                    # Sales data response
                    MagicMock(json=lambda: {"salesAndTrafficByAsin": [mock_amazon_sale]}),
                    # Advertising data response
                    MagicMock(json=lambda: {"campaigns": [mock_amazon_ad_data]})
                ]
                
                # Mock KSE client
                with patch('connectors.amazon_connector.KSEClient') as mock_kse:
                    mock_kse_instance = AsyncMock()
                    mock_kse.return_value = mock_kse_instance
                    mock_kse_instance.store_causal_memory.return_value = "memory_789"
                    
                    # Execute extraction
                    causal_data_list = await amazon_connector.extract_causal_marketing_data(
                        org_id="test_org",
                        start_date=date(2024, 1, 15),
                        end_date=date(2024, 1, 15),
                        historical_data=[]
                    )
                    
                    # Assertions
                    assert len(causal_data_list) == 1
                    causal_data = causal_data_list[0]
                    
                    assert isinstance(causal_data, CausalMarketingData)
                    assert causal_data.record_id == "123-4567890-1234567"
                    assert causal_data.treatment_type == TreatmentType.BUDGET_INCREASE
                    assert causal_data.outcome_value == 29.99
                    assert "marketplace_intelligence" in causal_data.causal_metadata
                    assert causal_data.causal_metadata["marketplace_intelligence"] == True
    
    @pytest.mark.asyncio
    async def test_marketplace_intelligence(self, amazon_connector, mock_amazon_sale):
        """Test marketplace intelligence integration"""
        from connectors.amazon_connector import AmazonSalesData
        
        sales_data = AmazonSalesData(**mock_amazon_sale)
        
        # Test buy box percentage impact
        assert sales_data.buy_box_percentage == 85.5
        assert sales_data.category_rank == 1250
        
        # Test customer segment determination
        segment = amazon_connector._determine_customer_segment(sales_data)
        assert segment == "regular_consumer"  # Not business, not high value
        
        # Test product category extraction
        category = amazon_connector._extract_product_category(sales_data)
        assert category == "general"  # Default category
    
    @pytest.mark.asyncio
    async def test_advertising_attribution(self, amazon_connector, mock_amazon_sale, mock_amazon_ad_data):
        """Test advertising attribution logic"""
        from connectors.amazon_connector import AmazonSalesData, AmazonAdvertisingData
        
        sales_data = AmazonSalesData(**mock_amazon_sale)
        ad_data = AmazonAdvertisingData(**mock_amazon_ad_data)
        
        treatment_result = await amazon_connector._determine_treatment_assignment(
            sales_data, ad_data, []
        )
        
        assert treatment_result.treatment_type == TreatmentType.BUDGET_INCREASE
        assert treatment_result.treatment_group == "sponsored_products"
        assert treatment_result.assignment_confidence == 0.9
        
        # Test ACOS categorization
        acos_category = amazon_connector._categorize_acos(ad_data.acos)
        assert acos_category == "highly_efficient"  # 8.4% ACOS


class TestKSEIntegration:
    """Test KSE Universal Substrate integration across all connectors"""
    
    @pytest.mark.asyncio
    async def test_neural_content_generation(self):
        """Test neural content generation for embeddings"""
        shopify_connector = ShopifyConnector("test.myshopify.com", "token")
        
        from connectors.shopify_connector import ShopifyOrderData
        order_data = ShopifyOrderData(
            id=12345,
            order_number="1001",
            created_at=datetime.now(),
            total_price=99.99,
            currency="USD",
            customer={"email": "test@example.com"},
            line_items=[{"title": "Test Product", "quantity": 1}]
        )
        
        neural_content = shopify_connector._create_neural_content(order_data, [], None)
        
        assert "Shopify order 12345" in neural_content
        assert "Test Product" in neural_content
        assert "$99.99" in neural_content
        assert len(neural_content) > 50  # Substantial content for embedding
    
    @pytest.mark.asyncio
    async def test_conceptual_space_mapping(self):
        """Test conceptual space dimension mapping"""
        woocommerce_connector = WooCommerceConnector("https://test.com", "key", "secret")
        
        from connectors.woocommerce_connector import WooCommerceOrderData
        order_data = WooCommerceOrderData(
            id=54321,
            number="2001",
            date_created=datetime.now(),
            status="completed",
            currency="USD",
            total=149.99,
            customer_id=98765
        )
        
        conceptual_space = woocommerce_connector._map_to_conceptual_space(order_data, None, None)
        
        assert "medium_value" in conceptual_space  # $149.99 is medium value
        assert "completed_order" in conceptual_space
        assert "_" in conceptual_space  # Multiple dimensions joined
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_nodes(self):
        """Test knowledge graph node creation"""
        amazon_connector = AmazonConnector(
            "MARKETPLACE", "SELLER", "KEY", "SECRET", "ARN", "CLIENT", "SECRET", "TOKEN"
        )
        
        from connectors.amazon_connector import AmazonSalesData
        sales_data = AmazonSalesData(
            order_id="123-456-789",
            purchase_date=datetime.now(),
            asin="B08N5WRWNW",
            sku="TEST-SKU",
            product_name="Test Product",
            quantity=1,
            item_price=29.99,
            currency="USD",
            marketplace_id="ATVPDKIKX0DER",
            fulfillment_channel="AFN"
        )
        
        from shared.models.causal_marketing import CausalMarketingData, TreatmentType, RandomizationUnit
        causal_data = CausalMarketingData(
            record_id="123-456-789",
            org_id="test_org",
            data_source="amazon_seller_central",
            treatment_type=TreatmentType.BUDGET_INCREASE,
            randomization_unit=RandomizationUnit.PRODUCT_CATEGORY,
            outcome_value=29.99
        )
        
        nodes = amazon_connector._create_knowledge_graph_nodes(sales_data, None, causal_data)
        
        assert "sale_123-456-789" in nodes
        assert "product_B08N5WRWNW" in nodes
        assert "sku_TEST-SKU" in nodes
        assert "marketplace_ATVPDKIKX0DER" in nodes
        assert "fulfillment_AFN" in nodes


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])