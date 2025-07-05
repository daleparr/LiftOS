"""
Test script for Marketing Data Endpoints in Memory Service
Tests the centralized data ingestion functionality
"""
import asyncio
import json
import httpx
from datetime import date, datetime
from typing import Dict, Any, List

# Test configuration
MEMORY_SERVICE_URL = "http://localhost:8003"
TEST_ORG_ID = "test_org_123"
TEST_USER_ID = "test_user_456"

# Test headers
TEST_HEADERS = {
    "X-User-Id": TEST_USER_ID,
    "X-Org-Id": TEST_ORG_ID,
    "X-Memory-Context": f"org_{TEST_ORG_ID}_context",
    "X-User-Roles": "admin,analyst",
    "Content-Type": "application/json"
}

# Sample marketing data for testing
SAMPLE_META_DATA = [
    {
        "id": "meta_campaign_001",
        "account_id": "act_123456789",
        "campaign_id": "camp_001",
        "campaign_name": "Holiday Sale Campaign",
        "spend": 1250.50,
        "impressions": 45000,
        "clicks": 890,
        "actions": [
            {"action_type": "purchase", "value": "23"},
            {"action_type": "add_to_cart", "value": "67"}
        ],
        "cpm": 27.79,
        "cpc": 1.40,
        "ctr": 1.98,
        "date_start": "2024-01-01",
        "date_stop": "2024-01-07"
    },
    {
        "id": "meta_campaign_002",
        "account_id": "act_123456789",
        "campaign_id": "camp_002",
        "campaign_name": "Brand Awareness Campaign",
        "spend": 890.25,
        "impressions": 67000,
        "clicks": 1200,
        "actions": [
            {"action_type": "purchase", "value": "15"},
            {"action_type": "add_to_cart", "value": "45"}
        ],
        "cpm": 13.29,
        "cpc": 0.74,
        "ctr": 1.79,
        "date_start": "2024-01-01",
        "date_stop": "2024-01-07"
    }
]

SAMPLE_GOOGLE_ADS_DATA = [
    {
        "id": "google_campaign_001",
        "customer_id": "123-456-7890",
        "campaign_id": "camp_google_001",
        "campaign_name": "Search Campaign - Electronics",
        "cost_micros": 2500000000,  # $2500 in micros
        "impressions": 78000,
        "clicks": 1560,
        "conversions": 45,
        "conversion_value": 4500.00,
        "date": "2024-01-01"
    },
    {
        "id": "google_campaign_002",
        "customer_id": "123-456-7890",
        "campaign_id": "camp_google_002",
        "campaign_name": "Display Campaign - Retargeting",
        "cost_micros": 1800000000,  # $1800 in micros
        "impressions": 120000,
        "clicks": 960,
        "conversions": 28,
        "conversion_value": 2800.00,
        "date": "2024-01-01"
    }
]

SAMPLE_KLAVIYO_DATA = [
    {
        "id": "klaviyo_campaign_001",
        "campaign_id": "klav_camp_001",
        "campaign_name": "Welcome Series Email",
        "list_id": "list_123",
        "delivered": 15000,
        "opened": 4500,
        "clicked": 675,
        "revenue": 3375.00,
        "unsubscribed": 12,
        "bounced": 45,
        "date": "2024-01-01"
    }
]


class MarketingEndpointTester:
    """Test class for marketing data endpoints"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = MEMORY_SERVICE_URL
        self.headers = TEST_HEADERS
    
    async def test_health_check(self) -> bool:
        """Test memory service health"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health Check: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health Check Error: {str(e)}")
            return False
    
    async def test_marketing_data_ingestion(self) -> bool:
        """Test marketing data ingestion endpoint"""
        print("\n🧪 Testing Marketing Data Ingestion...")
        
        # Test Meta Business data ingestion
        meta_payload = {
            "data_source": "meta_business",
            "data_entries": SAMPLE_META_DATA,
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-07",
            "metadata": {
                "test_batch": True,
                "ingestion_source": "api_test"
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/ingest",
                json=meta_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Meta Business Ingestion: {result['message']}")
                print(f"   Stored {len(result['data']['stored_entries'])} entries")
                return True
            else:
                print(f"❌ Meta Business Ingestion Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Meta Business Ingestion Error: {str(e)}")
            return False
    
    async def test_google_ads_ingestion(self) -> bool:
        """Test Google Ads data ingestion"""
        google_payload = {
            "data_source": "google_ads",
            "data_entries": SAMPLE_GOOGLE_ADS_DATA,
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-01",
            "metadata": {
                "test_batch": True,
                "ingestion_source": "api_test"
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/ingest",
                json=google_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Google Ads Ingestion: {result['message']}")
                print(f"   Stored {len(result['data']['stored_entries'])} entries")
                return True
            else:
                print(f"❌ Google Ads Ingestion Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Google Ads Ingestion Error: {str(e)}")
            return False
    
    async def test_klaviyo_ingestion(self) -> bool:
        """Test Klaviyo data ingestion"""
        klaviyo_payload = {
            "data_source": "klaviyo",
            "data_entries": SAMPLE_KLAVIYO_DATA,
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-01",
            "metadata": {
                "test_batch": True,
                "ingestion_source": "api_test"
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/ingest",
                json=klaviyo_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Klaviyo Ingestion: {result['message']}")
                print(f"   Stored {len(result['data']['stored_entries'])} entries")
                return True
            else:
                print(f"❌ Klaviyo Ingestion Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Klaviyo Ingestion Error: {str(e)}")
            return False
    
    async def test_marketing_search(self) -> bool:
        """Test marketing data search endpoint"""
        print("\n🔍 Testing Marketing Data Search...")
        
        search_payload = {
            "query": "marketing campaign data spend impressions",
            "data_sources": ["meta_business", "google_ads", "klaviyo"],
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-07",
            "limit": 10,
            "search_type": "hybrid"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/search",
                json=search_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Marketing Search: Found {result['data']['count']} results")
                if result['data']['results']:
                    print(f"   Sample result: {result['data']['results'][0]['metadata'].get('data_source', 'unknown')}")
                return True
            else:
                print(f"❌ Marketing Search Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Marketing Search Error: {str(e)}")
            return False
    
    async def test_marketing_insights(self) -> bool:
        """Test marketing insights generation"""
        print("\n📊 Testing Marketing Insights...")
        
        insights_payload = {
            "data_sources": ["meta_business", "google_ads", "klaviyo"],
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-07",
            "group_by": ["data_source", "campaign_id"],
            "metrics": ["spend", "impressions", "clicks", "conversions", "revenue"]
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/insights",
                json=insights_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                insights = result['data']
                print(f"✅ Marketing Insights Generated:")
                print(f"   Total Spend: ${insights.get('total_spend', 0):.2f}")
                print(f"   Total Impressions: {insights.get('total_impressions', 0):,}")
                print(f"   Total Clicks: {insights.get('total_clicks', 0):,}")
                print(f"   Average CPC: ${insights.get('average_cpc', 0):.2f}")
                print(f"   ROAS: {insights.get('roas', 0):.2f}")
                return True
            else:
                print(f"❌ Marketing Insights Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Marketing Insights Error: {str(e)}")
            return False
    
    async def test_calendar_dimensions(self) -> bool:
        """Test calendar dimensions endpoint"""
        print("\n📅 Testing Calendar Dimensions...")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/marketing/calendar/2024",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Calendar Dimensions: Generated {result['data']['total_days']} days for 2024")
                return True
            else:
                print(f"❌ Calendar Dimensions Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Calendar Dimensions Error: {str(e)}")
            return False
    
    async def test_causal_export(self) -> bool:
        """Test causal analysis export"""
        print("\n🔬 Testing Causal Analysis Export...")
        
        export_payload = {
            "data_sources": ["meta_business", "google_ads"],
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-07",
            "group_by": ["date", "data_source"],
            "metrics": ["spend", "impressions", "clicks", "conversions"]
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/export/causal?target_metric=conversions",
                json=export_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                causal_data = result['data']['causal_data']
                print(f"✅ Causal Export: {result['data']['records_count']} records exported")
                print(f"   Target Metric: {result['data']['target_metric']}")
                print(f"   Features: {len(causal_data.get('features', []))}")
                print(f"   Marketing Features: {causal_data.get('marketing_features', [])}")
                return True
            else:
                print(f"❌ Causal Export Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Causal Export Error: {str(e)}")
            return False
    
    async def test_data_transformation(self) -> bool:
        """Test data transformation endpoint"""
        print("\n🔄 Testing Data Transformation...")
        
        transform_payload = {
            "data_source": "meta_business",
            "data_entries": SAMPLE_META_DATA[:1],  # Test with one record
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-07"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/marketing/transform",
                json=transform_payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result['data']['transformation_summary']
                print(f"✅ Data Transformation:")
                print(f"   Original Records: {summary['original_records']}")
                print(f"   Transformed Records: {summary['transformed_records']}")
                print(f"   Columns Added: {summary['columns_added']}")
                return True
            else:
                print(f"❌ Data Transformation Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Data Transformation Error: {str(e)}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all marketing endpoint tests"""
        print("🚀 Starting Marketing Data Endpoints Test Suite")
        print("=" * 60)
        
        results = {}
        
        # Health check first
        results['health'] = await self.test_health_check()
        
        if not results['health']:
            print("❌ Memory service is not healthy. Skipping other tests.")
            return results
        
        # Data ingestion tests
        results['meta_ingestion'] = await self.test_marketing_data_ingestion()
        await asyncio.sleep(1)  # Brief pause between tests
        
        results['google_ingestion'] = await self.test_google_ads_ingestion()
        await asyncio.sleep(1)
        
        results['klaviyo_ingestion'] = await self.test_klaviyo_ingestion()
        await asyncio.sleep(2)  # Longer pause to allow indexing
        
        # Search and analytics tests
        results['search'] = await self.test_marketing_search()
        await asyncio.sleep(1)
        
        results['insights'] = await self.test_marketing_insights()
        await asyncio.sleep(1)
        
        results['calendar'] = await self.test_calendar_dimensions()
        await asyncio.sleep(1)
        
        results['causal_export'] = await self.test_causal_export()
        await asyncio.sleep(1)
        
        results['transformation'] = await self.test_data_transformation()
        
        return results
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    """Main test function"""
    tester = MarketingEndpointTester()
    
    try:
        results = await tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if passed_test:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("🎉 All marketing data endpoints are working correctly!")
        else:
            print("⚠️  Some tests failed. Check the Memory Service configuration.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {str(e)}")
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())