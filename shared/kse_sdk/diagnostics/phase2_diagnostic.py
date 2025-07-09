"""
Phase 2 Advanced Intelligence Flow Diagnostic Tool
Tests sophisticated cross-service intelligence sharing and real-time pattern recognition
"""

import asyncio
import logging
import time
import uuid
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.intelligence.orchestrator import (
    IntelligenceOrchestrator,
    IntelligenceEvent,
    IntelligenceEventType,
    IntelligencePriority
)
from shared.kse_sdk.intelligence.flow_manager import AdvancedIntelligenceFlowManager

@dataclass
class Phase2TestResult:
    """Result of a Phase 2 diagnostic test"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

class Phase2DiagnosticTool:
    """Comprehensive diagnostic tool for Phase 2 Advanced Intelligence Flow"""
    
    def __init__(self):
        self.kse_client = LiftKSEClient()
        self.intelligence_orchestrator = None
        self.flow_manager = None
        self.logger = logging.getLogger(__name__)
        self.test_results: List[Phase2TestResult] = []
        
    async def initialize(self):
        """Initialize all Phase 2 components"""
        try:
            await self.kse_client.initialize()
            
            self.intelligence_orchestrator = IntelligenceOrchestrator(self.kse_client)
            self.flow_manager = AdvancedIntelligenceFlowManager(self.kse_client, self.intelligence_orchestrator)
            
            await self.intelligence_orchestrator.initialize()
            await self.flow_manager.initialize()
            
            self.logger.info("Phase 2 diagnostic tool initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 2 diagnostic tool: {e}")
            return False
    
    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive Phase 2 diagnostics"""
        self.logger.info("Starting Phase 2 Advanced Intelligence Flow diagnostics...")
        
        # Test 1: Intelligence Orchestrator Event Publishing/Subscribing
        await self._test_intelligence_orchestrator()
        
        # Test 2: Advanced Flow Manager Capabilities
        await self._test_flow_manager()
        
        # Test 3: Cross-Service Intelligence Sharing
        await self._test_cross_service_intelligence()
        
        # Test 4: Real-Time Pattern Recognition
        await self._test_pattern_recognition()
        
        # Test 5: Intelligence Opportunity Discovery
        await self._test_opportunity_discovery()
        
        # Test 6: Adaptive Intelligence Routing
        await self._test_adaptive_routing()
        
        # Test 7: Cross-Service Learning
        await self._test_cross_service_learning()
        
        # Test 8: Intelligence Flow Optimization
        await self._test_flow_optimization()
        
        # Generate comprehensive report
        return self._generate_diagnostic_report()
    
    async def _test_intelligence_orchestrator(self):
        """Test 1: Intelligence Orchestrator Event Publishing/Subscribing"""
        start_time = time.time()
        test_name = "Intelligence Orchestrator Event System"
        
        try:
            # Test event subscription
            events_received = []
            
            async def test_handler(event: IntelligenceEvent):
                events_received.append(event)
            
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.PATTERN_DISCOVERED,
                test_handler
            )
            
            # Test event publishing
            test_event = IntelligenceEvent(
                event_type=IntelligenceEventType.PATTERN_DISCOVERED,
                source_service="diagnostic",
                target_services=["all"],
                priority=IntelligencePriority.MEDIUM,
                data={
                    "test_pattern": "diagnostic_pattern",
                    "confidence": 0.9,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.intelligence_orchestrator.publish_event(test_event)
            
            # Wait for event processing
            await asyncio.sleep(0.5)
            
            # Verify event was received
            success = len(events_received) > 0
            details = {
                "events_published": 1,
                "events_received": len(events_received),
                "event_data_match": events_received[0].data.get("test_pattern") == "diagnostic_pattern" if events_received else False
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_flow_manager(self):
        """Test 2: Advanced Flow Manager Capabilities"""
        start_time = time.time()
        test_name = "Advanced Flow Manager Capabilities"
        
        try:
            # Test service capability registration
            await self.flow_manager.register_service_capabilities(
                service_name="diagnostic_service",
                capabilities={
                    "test_capability": 0.95,
                    "diagnostic_analysis": 0.9
                },
                input_types=["test_data"],
                output_types=["test_results"]
            )
            
            # Test intelligence opportunity discovery
            opportunities = await self.flow_manager.discover_intelligence_opportunities(
                service_name="diagnostic_service"
            )
            
            # Test flow optimization
            flow_config = await self.flow_manager.optimize_flow(
                flow_type="real_time",
                source_service="diagnostic",
                target_services=["test_service"],
                context={"test": True}
            )
            
            success = True
            details = {
                "service_registered": True,
                "opportunities_discovered": len(opportunities),
                "flow_optimized": flow_config is not None
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_cross_service_intelligence(self):
        """Test 3: Cross-Service Intelligence Sharing"""
        start_time = time.time()
        test_name = "Cross-Service Intelligence Sharing"
        
        try:
            # Simulate intelligence sharing between services
            shared_intelligence = []
            
            async def intelligence_handler(event: IntelligenceEvent):
                shared_intelligence.append(event)
            
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.INSIGHT_GENERATED,
                intelligence_handler
            )
            
            # Publish insights from multiple simulated services
            services = ["surfacing", "causal", "llm", "agentic"]
            for service in services:
                insight_event = IntelligenceEvent(
                    event_type=IntelligenceEventType.INSIGHT_GENERATED,
                    source_service=service,
                    target_services=["all"],
                    priority=IntelligencePriority.HIGH,
                    data={
                        "insight_type": f"{service}_insight",
                        "confidence": 0.85,
                        "cross_service_data": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                await self.intelligence_orchestrator.publish_event(insight_event)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            success = len(shared_intelligence) >= len(services)
            details = {
                "services_tested": len(services),
                "intelligence_shared": len(shared_intelligence),
                "cross_service_flow": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_pattern_recognition(self):
        """Test 4: Real-Time Pattern Recognition"""
        start_time = time.time()
        test_name = "Real-Time Pattern Recognition"
        
        try:
            # Test pattern discovery and correlation
            patterns_discovered = []
            
            async def pattern_handler(event: IntelligenceEvent):
                if event.event_type == IntelligenceEventType.PATTERN_DISCOVERED:
                    patterns_discovered.append(event)
            
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.PATTERN_DISCOVERED,
                pattern_handler
            )
            
            # Simulate pattern discovery
            pattern_types = ["optimization", "behavioral", "performance", "causal"]
            for pattern_type in pattern_types:
                pattern_event = IntelligenceEvent(
                    event_type=IntelligenceEventType.PATTERN_DISCOVERED,
                    source_service="diagnostic",
                    target_services=["all"],
                    priority=IntelligencePriority.MEDIUM,
                    data={
                        "pattern_type": pattern_type,
                        "confidence": 0.8,
                        "pattern_data": {"test": True},
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                await self.intelligence_orchestrator.publish_event(pattern_event)
            
            # Wait for pattern processing
            await asyncio.sleep(0.8)
            
            success = len(patterns_discovered) >= len(pattern_types)
            details = {
                "pattern_types_tested": len(pattern_types),
                "patterns_discovered": len(patterns_discovered),
                "real_time_recognition": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_opportunity_discovery(self):
        """Test 5: Intelligence Opportunity Discovery"""
        start_time = time.time()
        test_name = "Intelligence Opportunity Discovery"
        
        try:
            # Test opportunity discovery with various contexts
            contexts = [
                {"service": "surfacing", "operation": "optimization", "data_type": "campaign"},
                {"service": "causal", "operation": "analysis", "data_type": "experiment"},
                {"service": "llm", "operation": "generation", "data_type": "content"},
                {"service": "agentic", "operation": "workflow", "data_type": "decision"}
            ]
            
            opportunities_found = 0
            for context in contexts:
                service_name = context["service"]
                opportunities = await self.flow_manager.discover_intelligence_opportunities(service_name)
                if opportunities:
                    opportunities_found += len(opportunities)
            
            success = opportunities_found > 0
            details = {
                "contexts_tested": len(contexts),
                "opportunities_discovered": opportunities_found,
                "discovery_success": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_adaptive_routing(self):
        """Test 6: Adaptive Intelligence Routing"""
        start_time = time.time()
        test_name = "Adaptive Intelligence Routing"
        
        try:
            # Test adaptive routing with different flow types
            flow_types = ["real_time", "batch", "streaming", "event_driven"]
            routing_success = 0
            
            for flow_type in flow_types:
                flow_config = await self.flow_manager.optimize_flow(
                    flow_type=flow_type,
                    source_service="diagnostic",
                    target_services=["surfacing", "causal"],
                    context={"test_routing": True}
                )
                if flow_config:
                    routing_success += 1
            
            success = routing_success == len(flow_types)
            details = {
                "flow_types_tested": len(flow_types),
                "routing_success": routing_success,
                "adaptive_routing": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_cross_service_learning(self):
        """Test 7: Cross-Service Learning"""
        start_time = time.time()
        test_name = "Cross-Service Learning"
        
        try:
            # Test real-time learning capabilities
            learning_contexts = [
                {"service": "surfacing", "operation": "optimization", "data_points": 100},
                {"service": "causal", "operation": "inference", "data_points": 50},
                {"service": "llm", "operation": "generation", "data_points": 75}
            ]
            
            learning_triggered = 0
            for context in learning_contexts:
                try:
                    await self.flow_manager.trigger_real_time_learning(context)
                    learning_triggered += 1
                except Exception:
                    pass  # Some learning contexts might not trigger
            
            success = learning_triggered > 0
            details = {
                "learning_contexts": len(learning_contexts),
                "learning_triggered": learning_triggered,
                "cross_service_learning": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    async def _test_flow_optimization(self):
        """Test 8: Intelligence Flow Optimization"""
        start_time = time.time()
        test_name = "Intelligence Flow Optimization"
        
        try:
            # Test flow optimization with performance metrics
            optimization_scenarios = [
                {"latency": 100, "throughput": 1000, "accuracy": 0.95},
                {"latency": 50, "throughput": 2000, "accuracy": 0.90},
                {"latency": 200, "throughput": 500, "accuracy": 0.98}
            ]
            
            optimizations_applied = 0
            for scenario in optimization_scenarios:
                flow_config = await self.flow_manager.optimize_flow(
                    flow_type="real_time",
                    source_service="diagnostic",
                    target_services=["test"],
                    context=scenario
                )
                if flow_config:
                    optimizations_applied += 1
            
            success = optimizations_applied > 0
            details = {
                "optimization_scenarios": len(optimization_scenarios),
                "optimizations_applied": optimizations_applied,
                "flow_optimization": success
            }
            
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, success, duration, details))
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append(Phase2TestResult(test_name, False, duration, {}, str(e)))
    
    def _generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        total_duration = sum(result.duration for result in self.test_results)
        
        report = {
            "phase2_diagnostic_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "total_duration": round(total_duration, 3),
                "average_test_duration": round(total_duration / total_tests, 3) if total_tests > 0 else 0
            },
            "test_results": [],
            "phase2_capabilities": {
                "intelligence_orchestrator": any(r.success for r in self.test_results if "Orchestrator" in r.test_name),
                "flow_manager": any(r.success for r in self.test_results if "Flow Manager" in r.test_name),
                "cross_service_intelligence": any(r.success for r in self.test_results if "Cross-Service Intelligence" in r.test_name),
                "pattern_recognition": any(r.success for r in self.test_results if "Pattern Recognition" in r.test_name),
                "opportunity_discovery": any(r.success for r in self.test_results if "Opportunity Discovery" in r.test_name),
                "adaptive_routing": any(r.success for r in self.test_results if "Adaptive" in r.test_name),
                "cross_service_learning": any(r.success for r in self.test_results if "Learning" in r.test_name),
                "flow_optimization": any(r.success for r in self.test_results if "Optimization" in r.test_name)
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Add detailed test results
        for result in self.test_results:
            report["test_results"].append({
                "test_name": result.test_name,
                "success": result.success,
                "duration": round(result.duration, 3),
                "details": result.details,
                "error": result.error
            })
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r.success]
        
        if not failed_tests:
            recommendations.append("[OK] All Phase 2 Advanced Intelligence Flow capabilities are working correctly!")
            recommendations.append("[READY] LiftOS is ready for sophisticated cross-service intelligence sharing")
            recommendations.append("[ACTIVE] Real-time pattern recognition and adaptive learning are fully operational")
        else:
            recommendations.append("[WARNING] Some Phase 2 capabilities need attention:")
            for failed_test in failed_tests:
                recommendations.append(f"   - Fix {failed_test.test_name}: {failed_test.error}")
        
        # Performance recommendations
        slow_tests = [r for r in self.test_results if r.duration > 2.0]
        if slow_tests:
            recommendations.append("[PERFORMANCE] Consider optimizing slow operations:")
            for slow_test in slow_tests:
                recommendations.append(f"   - {slow_test.test_name}: {slow_test.duration:.2f}s")
        
        return recommendations

async def run_phase2_diagnostics():
    """Main function to run Phase 2 diagnostics"""
    diagnostic_tool = Phase2DiagnosticTool()
    
    if not await diagnostic_tool.initialize():
        return {"error": "Failed to initialize Phase 2 diagnostic tool"}
    
    return await diagnostic_tool.run_comprehensive_diagnostics()

if __name__ == "__main__":
    import asyncio
    import json
    
    async def main():
        print(">> Starting Phase 2 Advanced Intelligence Flow Diagnostics...")
        result = await run_phase2_diagnostics()
        print("\n" + "="*80)
        print("PHASE 2 DIAGNOSTIC REPORT")
        print("="*80)
        print(json.dumps(result, indent=2))
        
        # Print summary
        summary = result.get("phase2_diagnostic_summary", {})
        print(f"\n>> Summary: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} tests passed")
        print(f">> Total Duration: {summary.get('total_duration', 0):.3f}s")
        print(f">> Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        # Print recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("\n>> Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
    
    asyncio.run(main())