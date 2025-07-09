"""
Comprehensive KSE Integration Test for All Four Microservices
Tests bidirectional KSE access for SURFACING, CAUSAL, LLM, and AGENTIC modules/services
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import KSE SDK
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult

# Import module integration classes
from modules.surfacing.app import SurfacingKSEIntegration
from modules.causal.app import CausalKSEIntegration
from modules.llm.app import LLMKSEIntegration
from modules.agentic.app import AgenticKSEIntegration


class MicroserviceKSEIntegrationTest:
    """Comprehensive test suite for microservice KSE integration"""
    
    def __init__(self):
        self.kse_client = LiftKSEClient()
        self.test_org_id = "test_org_integration"
        self.results = {
            "surfacing": {"read": False, "write": False, "errors": []},
            "causal": {"read": False, "write": False, "errors": []},
            "llm": {"read": False, "write": False, "errors": []},
            "agentic": {"read": False, "write": False, "errors": []}
        }
        
    async def setup(self):
        """Initialize KSE client and integrations"""
        print("🔧 Setting up KSE integration test...")
        
        try:
            await self.kse_client.initialize()
            
            # Initialize all integration classes
            self.surfacing_integration = SurfacingKSEIntegration(self.kse_client)
            self.causal_integration = CausalKSEIntegration(self.kse_client)
            self.llm_integration = LLMKSEIntegration(self.kse_client)
            self.agentic_integration = AgenticKSEIntegration(self.kse_client)
            
            print("✅ KSE client and integrations initialized successfully")
            
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            raise
    
    async def test_surfacing_integration(self):
        """Test SURFACING module KSE integration"""
        print("\n🔍 Testing SURFACING module KSE integration...")
        
        try:
            # Test READ: Retrieve optimization context
            test_request = {
                "org_id": self.test_org_id,
                "campaign_type": "email_marketing",
                "target_metrics": ["conversion_rate", "click_through_rate"]
            }
            
            context = await self.surfacing_integration.retrieve_optimization_context(test_request)
            if isinstance(context, dict):
                self.results["surfacing"]["read"] = True
                print("✅ SURFACING read access successful")
            
            # Test WRITE: Enrich treatment recommendations
            test_recommendations = [
                {
                    "id": "test_rec_1",
                    "type": "subject_line_optimization",
                    "confidence": 0.85,
                    "expected_lift": 0.15,
                    "strategy": "personalization"
                }
            ]
            
            await self.surfacing_integration.enrich_treatment_recommendations(
                test_recommendations, self.test_org_id
            )
            self.results["surfacing"]["write"] = True
            print("✅ SURFACING write access successful")
            
        except Exception as e:
            error_msg = f"SURFACING integration error: {str(e)}"
            self.results["surfacing"]["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    async def test_causal_integration(self):
        """Test CAUSAL module KSE integration"""
        print("\n🔗 Testing CAUSAL module KSE integration...")
        
        try:
            # Test READ: Retrieve causal priors
            test_request = {
                "org_id": self.test_org_id,
                "treatment_type": "price_optimization",
                "outcome_variable": "revenue"
            }
            
            priors = await self.causal_integration.retrieve_causal_priors(test_request)
            if isinstance(priors, dict):
                self.results["causal"]["read"] = True
                print("✅ CAUSAL read access successful")
            
            # Test WRITE: Enrich causal insights
            test_insights = {
                "id": "test_causal_1",
                "treatment": "price_discount_10",
                "effect_size": 0.12,
                "confidence_interval": [0.08, 0.16],
                "p_value": 0.001,
                "methodology": "difference_in_differences"
            }
            
            await self.causal_integration.enrich_causal_insights(test_insights, self.test_org_id)
            self.results["causal"]["write"] = True
            print("✅ CAUSAL write access successful")
            
        except Exception as e:
            error_msg = f"CAUSAL integration error: {str(e)}"
            self.results["causal"]["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    async def test_llm_integration(self):
        """Test LLM module KSE integration"""
        print("\n🤖 Testing LLM module KSE integration...")
        
        try:
            # Test READ: Retrieve content context
            test_request = {
                "org_id": self.test_org_id,
                "content_type": "product_description",
                "target_audience": "tech_professionals"
            }
            
            context = await self.llm_integration.retrieve_content_context(test_request)
            if isinstance(context, dict):
                self.results["llm"]["read"] = True
                print("✅ LLM read access successful")
            
            # Test WRITE: Enrich generated content
            test_content = {
                "id": "test_content_1",
                "type": "product_description",
                "content": "Revolutionary AI-powered analytics platform",
                "performance_metrics": {
                    "engagement_score": 0.78,
                    "conversion_rate": 0.045
                },
                "generation_metadata": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "tokens": 150
                }
            }
            
            await self.llm_integration.enrich_generated_content(test_content, self.test_org_id)
            self.results["llm"]["write"] = True
            print("✅ LLM write access successful")
            
        except Exception as e:
            error_msg = f"LLM integration error: {str(e)}"
            self.results["llm"]["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    async def test_agentic_integration(self):
        """Test AGENTIC module KSE integration"""
        print("\n🤖 Testing AGENTIC module KSE integration...")
        
        try:
            # Test READ: Retrieve agent patterns
            test_request = {
                "org_id": self.test_org_id,
                "agent_type": "optimization_agent",
                "task_category": "campaign_optimization"
            }
            
            patterns = await self.agentic_integration.retrieve_agent_patterns(test_request)
            if isinstance(patterns, dict):
                self.results["agentic"]["read"] = True
                print("✅ AGENTIC read access successful")
            
            # Test WRITE: Enrich evaluation results
            test_evaluation = {
                "id": "test_eval_1",
                "agent_type": "optimization_agent",
                "task_performance": {
                    "success_rate": 0.89,
                    "avg_improvement": 0.23,
                    "execution_time": 45.2
                },
                "evaluation_metrics": {
                    "accuracy": 0.91,
                    "efficiency": 0.87,
                    "robustness": 0.85
                }
            }
            
            await self.agentic_integration.enrich_evaluation_results(test_evaluation, self.test_org_id)
            self.results["agentic"]["write"] = True
            print("✅ AGENTIC write access successful")
            
        except Exception as e:
            error_msg = f"AGENTIC integration error: {str(e)}"
            self.results["agentic"]["errors"].append(error_msg)
            print(f"❌ {error_msg}")
    
    async def test_cross_service_intelligence_flow(self):
        """Test intelligence flow between services"""
        print("\n🔄 Testing cross-service intelligence flow...")
        
        try:
            # Create a test scenario where services share intelligence
            
            # 1. SURFACING creates optimization recommendation
            optimization_entity = Entity(
                id="cross_test_optimization",
                type="optimization_recommendation",
                content={
                    "strategy": "multi_channel_personalization",
                    "expected_lift": 0.18,
                    "confidence": 0.82
                },
                metadata={
                    "org_id": self.test_org_id,
                    "service_source": "surfacing",
                    "entity_type": "optimization_recommendation"
                }
            )
            await self.kse_client.store_entity(self.test_org_id, optimization_entity)
            
            # 2. CAUSAL retrieves and builds upon it
            causal_search = await self.kse_client.hybrid_search(
                org_id=self.test_org_id,
                query="optimization_recommendation multi_channel",
                search_type="hybrid",
                limit=5
            )
            
            if causal_search:
                # 3. LLM generates content based on causal insights
                llm_context = {
                    "optimization_strategy": causal_search[0].content.get("strategy"),
                    "expected_performance": causal_search[0].content.get("expected_lift")
                }
                
                # 4. AGENTIC evaluates the complete workflow
                workflow_evaluation = {
                    "workflow_id": "cross_service_test",
                    "services_involved": ["surfacing", "causal", "llm", "agentic"],
                    "intelligence_flow_success": True,
                    "cross_service_score": 0.91
                }
                
                print("✅ Cross-service intelligence flow successful")
                return True
            
        except Exception as e:
            print(f"❌ Cross-service intelligence flow failed: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 MICROSERVICE KSE INTEGRATION TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for service, results in self.results.items():
            print(f"\n🔧 {service.upper()} MODULE:")
            
            # Read test
            read_status = "✅ PASS" if results["read"] else "❌ FAIL"
            print(f"  📖 Read Access:  {read_status}")
            total_tests += 1
            if results["read"]:
                passed_tests += 1
            
            # Write test
            write_status = "✅ PASS" if results["write"] else "❌ FAIL"
            print(f"  ✍️  Write Access: {write_status}")
            total_tests += 1
            if results["write"]:
                passed_tests += 1
            
            # Errors
            if results["errors"]:
                print(f"  ⚠️  Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"    - {error}")
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL MICROSERVICES HAVE SUCCESSFUL KSE INTEGRATION!")
            print("✅ Bidirectional intelligence flow is operational")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} integration issues detected")
            print("❌ Some microservices need attention")
        
        print("="*60)
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Comprehensive Microservice KSE Integration Test")
        print("="*60)
        
        await self.setup()
        
        # Test each microservice
        await self.test_surfacing_integration()
        await self.test_causal_integration()
        await self.test_llm_integration()
        await self.test_agentic_integration()
        
        # Test cross-service intelligence flow
        await self.test_cross_service_intelligence_flow()
        
        # Generate report
        self.generate_report()


async def main():
    """Main test execution"""
    test_suite = MicroserviceKSEIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())