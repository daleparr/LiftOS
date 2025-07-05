#!/usr/bin/env python3
"""
LiftOS Complete Integration Validation
Tests Surfacing, Causal AI, and LLM modules working together in the LiftOS ecosystem.
"""

import requests
import json
import time
import sys
from datetime import datetime
import asyncio
import concurrent.futures

# Configuration
SERVICES = {
    "auth": "http://localhost:8001",
    "registry": "http://localhost:8005",
    "memory": "http://localhost:8003",
    "gateway": "http://localhost:8000",
    "surfacing_service": "http://localhost:3002",
    "surfacing_module": "http://localhost:8007",
    "causal_service": "http://localhost:3003",
    "causal_module": "http://localhost:8008",
    "llm_service": "http://localhost:3004",
    "llm_module": "http://localhost:8009"
}

# Test data for integrated workflow
INTEGRATED_TEST_DATA = {
    "text_content": """
    Our Q3 marketing campaign generated exceptional results across multiple channels. 
    Google Ads delivered a 15% increase in conversions with a 3.2x ROAS. 
    Meta campaigns showed strong brand awareness lift of 22% among target demographics.
    TikTok ads performed surprisingly well with 18% conversion rate improvement.
    Email marketing maintained steady performance with 4.1% CTR.
    However, we noticed declining organic reach and increased competition in paid search.
    Customer sentiment analysis reveals 85% positive feedback on new product features.
    """,
    "campaign_data": {
        "campaigns": [
            {
                "id": "q3_google",
                "name": "Q3 Google Ads",
                "platform": "google_ads",
                "spend": 25000,
                "conversions": 1250,
                "revenue": 80000
            },
            {
                "id": "q3_meta",
                "name": "Q3 Meta Campaign",
                "platform": "meta",
                "spend": 18000,
                "conversions": 720,
                "revenue": 45000
            }
        ]
    }
}

class IntegrationValidator:
    def __init__(self):
        self.auth_token = None
        self.test_results = []
        self.surfacing_results = None
        self.causal_results = None
        
    def log_test(self, test_name, success, message="", details=None):
        """Log test result with details"""
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        if details and not success:
            print(f"    Details: {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        print()

    def test_all_services_health(self):
        """Test health of all services"""
        print("Testing service health...")
        
        for service_name, url in SERVICES.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                success = response.status_code == 200
                self.log_test(f"{service_name.title()} Health", success,
                             "Service is healthy" if success else f"Status: {response.status_code}")
            except Exception as e:
                self.log_test(f"{service_name.title()} Health", False, f"Connection failed: {e}")

    def test_module_registrations(self):
        """Test that all modules are registered"""
        print("Testing module registrations...")
        
        modules = ["surfacing", "causal", "llm"]
        for module in modules:
            try:
                response = requests.get(f"{SERVICES['registry']}/modules/{module}", timeout=5)
                if response.status_code == 200:
                    module_data = response.json()
                    success = module_data.get("status") == "active"
                    self.log_test(f"{module.title()} Module Registration", success,
                                 f"Status: {module_data.get('status', 'unknown')}")
                else:
                    self.log_test(f"{module.title()} Module Registration", False,
                                 f"Module not found: {response.status_code}")
            except Exception as e:
                self.log_test(f"{module.title()} Module Registration", False, f"Error: {e}")

    def test_surfacing_analysis(self):
        """Test surfacing analysis and store results"""
        print("Testing surfacing analysis...")
        
        try:
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{SERVICES['surfacing_module']}/analyze",
                json={"text": INTEGRATED_TEST_DATA["text_content"]},
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.surfacing_results = response.json()
                has_sentiment = "sentiment" in self.surfacing_results
                has_entities = "entities" in self.surfacing_results
                has_insights = "insights" in self.surfacing_results
                
                success = has_sentiment and has_entities and has_insights
                self.log_test("Surfacing Analysis", success,
                             f"Extracted {len(self.surfacing_results.get('entities', []))} entities, "
                             f"sentiment: {self.surfacing_results.get('sentiment', {}).get('label', 'unknown')}")
            else:
                self.log_test("Surfacing Analysis", False,
                             f"Request failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Surfacing Analysis", False, f"Error: {e}")

    def test_causal_analysis(self):
        """Test causal analysis and store results"""
        print("Testing causal analysis...")
        
        try:
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{SERVICES['causal_module']}/attribution/analyze",
                json=INTEGRATED_TEST_DATA["campaign_data"],
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                self.causal_results = response.json()
                has_attribution = "attribution_scores" in self.causal_results
                has_insights = "insights" in self.causal_results
                
                success = has_attribution and has_insights
                self.log_test("Causal Analysis", success,
                             f"Generated attribution analysis with insights")
            else:
                self.log_test("Causal Analysis", False,
                             f"Request failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Causal Analysis", False, f"Error: {e}")

    def test_llm_analysis(self):
        """Test LLM analysis and store results"""
        print("Testing LLM analysis...")
        
        try:
            headers = {"Content-Type": "application/json"}
            
            # Test content generation
            generation_request = {
                "template": "ad_copy",
                "variables": {
                    "product": "wireless headphones",
                    "audience": "music enthusiasts",
                    "benefits": "superior sound quality and noise cancellation"
                },
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "max_tokens": 150
            }
            
            response = requests.post(
                f"{SERVICES['llm_module']}/api/v1/prompts/generate",
                json=generation_request,
                headers=headers,
                timeout=45
            )
            
            if response.status_code == 200:
                self.llm_results = response.json()
                has_content = "generated_content" in self.llm_results
                has_metadata = "metadata" in self.llm_results
                
                success = has_content and has_metadata
                self.log_test("LLM Analysis", success,
                             f"Generated content using {self.llm_results.get('metadata', {}).get('provider', 'unknown')}")
            else:
                self.log_test("LLM Analysis", False,
                             f"Request failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("LLM Analysis", False, f"Error: {e}")

    def test_cross_module_insights(self):
        """Test combining insights from all three modules"""
        print("Testing cross-module insights integration...")
        
        if not self.surfacing_results or not self.causal_results:
            self.log_test("Cross-Module Integration", False,
                         "Cannot test integration - missing analysis results")
            return
        
        try:
            # Combine insights from all three modules
            combined_insights = {
                "text_analysis": {
                    "sentiment": self.surfacing_results.get("sentiment"),
                    "key_entities": self.surfacing_results.get("entities", [])[:5],
                    "insights": self.surfacing_results.get("insights", [])
                },
                "attribution_analysis": {
                    "top_channels": self.causal_results.get("attribution_scores", {}),
                    "recommendations": self.causal_results.get("insights", [])
                },
                "llm_generation": {
                    "generated_content": getattr(self, 'llm_results', {}).get("generated_content", ""),
                    "provider": getattr(self, 'llm_results', {}).get("metadata", {}).get("provider", ""),
                    "tokens_used": getattr(self, 'llm_results', {}).get("metadata", {}).get("tokens_used", 0)
                },
                "integrated_recommendations": []
            }
            
            # Generate integrated recommendations
            sentiment_score = self.surfacing_results.get("sentiment", {}).get("score", 0)
            if sentiment_score > 0.7:
                combined_insights["integrated_recommendations"].append(
                    "Positive sentiment detected - consider increasing budget for top-performing channels"
                )
            
            # Add LLM-based recommendations
            if hasattr(self, 'llm_results') and self.llm_results:
                combined_insights["integrated_recommendations"].append(
                    "Generated optimized content using LLM - ready for campaign deployment"
                )
            
            # Test storing combined insights in memory
            headers = {"Content-Type": "application/json"}
            memory_data = {
                "analysis_type": "integrated_marketing_analysis",
                "results": combined_insights,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "modules_used": ["surfacing", "causal", "llm"]
                }
            }
            
            response = requests.post(
                f"{SERVICES['memory']}/store",
                json=memory_data,
                headers=headers,
                timeout=10
            )
            
            success = response.status_code in [200, 201]
            self.log_test("Cross-Module Integration", success,
                         f"Successfully combined and stored integrated insights" if success
                         else f"Failed to store combined insights: {response.status_code}")
            
        except Exception as e:
            self.log_test("Cross-Module Integration", False, f"Error: {e}")

    def test_api_gateway_routing(self):
        """Test API gateway routing to both modules"""
        print("Testing API gateway routing...")
        
        # Test routing to surfacing module
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.get(f"{SERVICES['gateway']}/modules/surfacing/health", 
                                  headers=headers, timeout=5)
            success = response.status_code == 200
            self.log_test("Gateway Surfacing Routing", success,
                         "Gateway correctly routes to surfacing module" if success
                         else f"Routing failed: {response.status_code}")
        except Exception as e:
            self.log_test("Gateway Surfacing Routing", False, f"Error: {e}")
        
        # Test routing to causal module
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.get(f"{SERVICES['gateway']}/modules/causal/health",
                                  headers=headers, timeout=5)
            success = response.status_code == 200
            self.log_test("Gateway Causal Routing", success,
                         "Gateway correctly routes to causal module" if success
                         else f"Routing failed: {response.status_code}")
        except Exception as e:
            self.log_test("Gateway Causal Routing", False, f"Error: {e}")
        
        # Test routing to LLM module
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.get(f"{SERVICES['gateway']}/modules/llm/health",
                                  headers=headers, timeout=5)
            success = response.status_code == 200
            self.log_test("Gateway LLM Routing", success,
                         "Gateway correctly routes to LLM module" if success
                         else f"Routing failed: {response.status_code}")
        except Exception as e:
            self.log_test("Gateway LLM Routing", False, f"Error: {e}")

    def test_memory_persistence(self):
        """Test that analysis results persist in memory"""
        print("Testing memory persistence...")
        
        try:
            # Query stored analyses
            response = requests.get(f"{SERVICES['memory']}/query", 
                                  params={"type": "integrated_marketing_analysis"},
                                  timeout=10)
            
            if response.status_code == 200:
                stored_data = response.json()
                has_data = len(stored_data.get("results", [])) > 0
                self.log_test("Memory Persistence", has_data,
                             f"Found {len(stored_data.get('results', []))} stored analyses" if has_data
                             else "No stored analyses found")
            else:
                self.log_test("Memory Persistence", False,
                             f"Query failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Memory Persistence", False, f"Error: {e}")

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with LLM integration"""
        print("Testing end-to-end workflow...")
        
        try:
            # Step 1: Analyze text content
            text_response = requests.post(
                f"{SERVICES['surfacing_module']}/analyze",
                json={"text": INTEGRATED_TEST_DATA["text_content"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Step 2: Analyze campaign attribution
            attribution_response = requests.post(
                f"{SERVICES['causal_module']}/attribution/analyze",
                json=INTEGRATED_TEST_DATA["campaign_data"],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Step 3: Generate LLM content based on insights
            if text_response.status_code == 200 and attribution_response.status_code == 200:
                text_results = text_response.json()
                attribution_results = attribution_response.json()
                
                # Generate marketing recommendations using LLM
                llm_data = {
                    "prompt": f"Generate marketing recommendations based on sentiment analysis and attribution data",
                    "template": "marketing_recommendations",
                    "context": {
                        "sentiment": text_results.get("sentiment", {}),
                        "attribution": attribution_results.get("attribution_scores", {}),
                        "campaign_data": INTEGRATED_TEST_DATA["campaign_data"]
                    }
                }
                
                llm_response = requests.post(
                    f"{SERVICES['llm_module']}/generate",
                    json=llm_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                # Step 4: Generate budget optimization based on all insights
                budget_data = {
                    "total_budget": 50000,
                    "channels": ["google_ads", "meta", "tiktok"],
                    "sentiment_boost": text_results.get("sentiment", {}).get("score", 0),
                    "attribution_weights": attribution_results.get("attribution_scores", {}),
                    "llm_recommendations": llm_response.json().get("content", "") if llm_response.status_code == 200 else ""
                }
                
                budget_response = requests.post(
                    f"{SERVICES['causal_module']}/budget/optimize",
                    json=budget_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                success = (text_response.status_code == 200 and
                          attribution_response.status_code == 200 and
                          llm_response.status_code == 200 and
                          budget_response.status_code == 200)
                
                self.log_test("End-to-End Workflow", success,
                             "Complete workflow with LLM integration executed successfully" if success
                             else "Workflow failed at one or more steps")
            else:
                self.log_test("End-to-End Workflow", False,
                             "Initial analysis steps failed")
                
        except Exception as e:
            self.log_test("End-to-End Workflow", False, f"Error: {e}")

    def run_validation(self):
        """Run complete validation suite"""
        print("========================================")
        print("LiftOS Complete Integration Validation")
        print("========================================")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run validation tests
        self.test_all_services_health()
        self.test_module_registrations()
        self.test_surfacing_analysis()
        self.test_causal_analysis()
        self.test_llm_analysis()
        self.test_cross_module_insights()
        self.test_api_gateway_routing()
        self.test_memory_persistence()
        self.test_end_to_end_workflow()
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("========================================")
        print("Integration Validation Summary")
        print("========================================")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        if failed_tests > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
            print()
        else:
            print("🎉 All integration tests passed!")
            print("Both Surfacing and Causal AI modules are fully integrated with LiftOS!")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return failed_tests == 0

def main():
    """Main validation execution"""
    validator = IntegrationValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()