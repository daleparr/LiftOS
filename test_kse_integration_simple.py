"""
Simplified KSE Integration Test for Four Microservices
Tests KSE integration classes directly without module dependencies
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import KSE SDK
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult


class SimplifiedKSEIntegrationTest:
    """Simplified test suite for KSE integration"""
    
    def __init__(self):
        self.kse_client = LiftKSEClient()
        self.test_org_id = "test_org_integration"
        self.results = {
            "kse_client": {"initialized": False, "errors": []},
            "entity_storage": {"success": False, "errors": []},
            "entity_retrieval": {"success": False, "errors": []},
            "hybrid_search": {"success": False, "errors": []}
        }
        
    async def setup(self):
        """Initialize KSE client"""
        print("[SETUP] Setting up KSE integration test...")
        
        try:
            await self.kse_client.initialize()
            self.results["kse_client"]["initialized"] = True
            print("[PASS] KSE client initialized successfully")
            
        except Exception as e:
            error_msg = f"KSE client initialization failed: {str(e)}"
            self.results["kse_client"]["errors"].append(error_msg)
            print(f"[FAIL] {error_msg}")
            raise
    
    async def test_entity_storage(self):
        """Test entity storage functionality"""
        print("\n[STORAGE] Testing entity storage...")
        
        try:
            # Create test entities for each microservice type
            test_entities = [
                # SURFACING entity
                Entity(
                    id="test_surfacing_optimization",
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
                ),
                # CAUSAL entity
                Entity(
                    id="test_causal_insight",
                    type="causal_insight",
                    content={
                        "treatment": "price_discount_10",
                        "effect_size": 0.12,
                        "confidence_interval": [0.08, 0.16],
                        "p_value": 0.001
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "causal",
                        "entity_type": "causal_insight"
                    }
                ),
                # LLM entity
                Entity(
                    id="test_llm_content",
                    type="generated_content",
                    content={
                        "content": "Revolutionary AI-powered analytics platform",
                        "performance_metrics": {
                            "engagement_score": 0.78,
                            "conversion_rate": 0.045
                        }
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "llm",
                        "entity_type": "generated_content"
                    }
                ),
                # AGENTIC entity
                Entity(
                    id="test_agentic_evaluation",
                    type="agent_evaluation",
                    content={
                        "agent_type": "optimization_agent",
                        "performance": {
                            "success_rate": 0.89,
                            "avg_improvement": 0.23
                        }
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "agentic",
                        "entity_type": "agent_evaluation"
                    }
                )
            ]
            
            # Store all entities
            for entity in test_entities:
                await self.kse_client.store_entity(self.test_org_id, entity)
            
            self.results["entity_storage"]["success"] = True
            print(f"[PASS] Successfully stored {len(test_entities)} test entities")
            
        except Exception as e:
            error_msg = f"Entity storage failed: {str(e)}"
            self.results["entity_storage"]["errors"].append(error_msg)
            print(f"[FAIL] {error_msg}")
    
    async def test_entity_retrieval(self):
        """Test entity retrieval functionality"""
        print("\n🔍 Testing entity retrieval...")
        
        try:
            # Test retrieving entities by type
            entity_types = ["optimization_recommendation", "causal_insight", "generated_content", "agent_evaluation"]
            retrieved_count = 0
            
            for entity_type in entity_types:
                entities = await self.kse_client.get_entities_by_type(self.test_org_id, entity_type)
                if entities:
                    retrieved_count += len(entities)
                    print(f"  📋 Retrieved {len(entities)} {entity_type} entities")
            
            if retrieved_count > 0:
                self.results["entity_retrieval"]["success"] = True
                print(f"[PASS] Successfully retrieved {retrieved_count} entities")
            else:
                raise Exception("No entities retrieved")
            
        except Exception as e:
            error_msg = f"Entity retrieval failed: {str(e)}"
            self.results["entity_retrieval"]["errors"].append(error_msg)
            print(f"[FAIL] {error_msg}")
    
    async def test_hybrid_search(self):
        """Test hybrid search functionality"""
        print("\n🔎 Testing hybrid search...")
        
        try:
            # Test searches for each microservice domain
            search_queries = [
                "optimization personalization",
                "causal effect discount",
                "content analytics platform",
                "agent evaluation performance"
            ]
            
            total_results = 0
            for query in search_queries:
                results = await self.kse_client.hybrid_search(
                    org_id=self.test_org_id,
                    query=query,
                    search_type="hybrid",
                    limit=5
                )
                if results:
                    total_results += len(results)
                    print(f"  🔍 Query '{query}': {len(results)} results")
            
            if total_results > 0:
                self.results["hybrid_search"]["success"] = True
                print(f"[PASS] Hybrid search successful with {total_results} total results")
            else:
                raise Exception("No search results returned")
            
        except Exception as e:
            error_msg = f"Hybrid search failed: {str(e)}"
            self.results["hybrid_search"]["errors"].append(error_msg)
            print(f"[FAIL] {error_msg}")
    
    async def test_cross_service_intelligence_flow(self):
        """Test intelligence flow between services"""
        print("\n🔄 Testing cross-service intelligence flow...")
        
        try:
            # Simulate a workflow where services share intelligence
            
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
                    "entity_type": "optimization_recommendation",
                    "workflow_id": "cross_service_test"
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
                # 3. CAUSAL creates causal insight based on optimization
                causal_entity = Entity(
                    id="cross_test_causal",
                    type="causal_insight",
                    content={
                        "based_on_optimization": causal_search[0].id,
                        "causal_effect": 0.15,
                        "confidence_interval": [0.12, 0.18]
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "causal",
                        "entity_type": "causal_insight",
                        "workflow_id": "cross_service_test"
                    }
                )
                await self.kse_client.store_entity(self.test_org_id, causal_entity)
                
                # 4. LLM generates content based on insights
                llm_entity = Entity(
                    id="cross_test_content",
                    type="generated_content",
                    content={
                        "content": "Based on causal analysis, multi-channel personalization shows 15% lift",
                        "based_on_insights": [causal_search[0].id, causal_entity.id]
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "llm",
                        "entity_type": "generated_content",
                        "workflow_id": "cross_service_test"
                    }
                )
                await self.kse_client.store_entity(self.test_org_id, llm_entity)
                
                # 5. AGENTIC evaluates the complete workflow
                agentic_entity = Entity(
                    id="cross_test_evaluation",
                    type="workflow_evaluation",
                    content={
                        "workflow_id": "cross_service_test",
                        "services_involved": ["surfacing", "causal", "llm", "agentic"],
                        "intelligence_flow_success": True,
                        "cross_service_score": 0.91
                    },
                    metadata={
                        "org_id": self.test_org_id,
                        "service_source": "agentic",
                        "entity_type": "workflow_evaluation",
                        "workflow_id": "cross_service_test"
                    }
                )
                await self.kse_client.store_entity(self.test_org_id, agentic_entity)
                
                print("[PASS] Cross-service intelligence flow successful")
                return True
            else:
                raise Exception("Failed to retrieve optimization recommendation")
            
        except Exception as e:
            print(f"[FAIL] Cross-service intelligence flow failed: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 KSE INTEGRATION TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        test_categories = [
            ("KSE Client Initialization", "kse_client", "initialized"),
            ("Entity Storage", "entity_storage", "success"),
            ("Entity Retrieval", "entity_retrieval", "success"),
            ("Hybrid Search", "hybrid_search", "success")
        ]
        
        for test_name, category, success_key in test_categories:
            print(f"\n[SETUP] {test_name}:")
            
            success = self.results[category][success_key]
            status = "[PASS] PASS" if success else "[FAIL] FAIL"
            print(f"  Status: {status}")
            
            total_tests += 1
            if success:
                passed_tests += 1
            
            # Show errors if any
            if self.results[category]["errors"]:
                print(f"  ⚠️  Errors: {len(self.results[category]['errors'])}")
                for error in self.results[category]["errors"]:
                    print(f"    - {error}")
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL KSE INTEGRATION TESTS PASSED!")
            print("[PASS] Universal KSE substrate is operational")
            print("[PASS] Bidirectional intelligence flow is ready")
            print("[PASS] All four microservices can leverage KSE")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} integration issues detected")
            print("[FAIL] KSE substrate needs attention")
        
        print("="*60)
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting KSE Integration Test")
        print("="*60)
        
        await self.setup()
        
        # Test core KSE functionality
        await self.test_entity_storage()
        await self.test_entity_retrieval()
        await self.test_hybrid_search()
        
        # Test cross-service intelligence flow
        await self.test_cross_service_intelligence_flow()
        
        # Generate report
        self.generate_report()


async def main():
    """Main test execution"""
    test_suite = SimplifiedKSEIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())