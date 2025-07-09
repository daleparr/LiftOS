"""
LiftOS Strategic Intelligence Service
Market intelligence, competitive analysis, and strategic insights.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from collections import defaultdict
import statistics
import json

# Import models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.business import (
    CompetitiveIntelligence, MarketIntelligence, BusinessMetric,
    MetricType, MetricFrequency
)
from shared.kse_sdk.client import LiftKSEClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategicTrend:
    """Strategic trend analysis"""
    def __init__(self, trend_id: str, trend_data: Dict[str, Any]):
        self.id = trend_id
        self.name = trend_data.get('name', '')
        self.description = trend_data.get('description', '')
        self.trend_type = trend_data.get('trend_type', 'market')  # market, technology, competitive
        self.impact_level = trend_data.get('impact_level', 'medium')  # low, medium, high
        self.time_horizon = trend_data.get('time_horizon', 'medium')  # short, medium, long
        self.confidence_score = trend_data.get('confidence_score', 0.5)
        self.data_points: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()

class OpportunityAssessment:
    """Business opportunity assessment"""
    def __init__(self, opportunity_id: str, opportunity_data: Dict[str, Any]):
        self.id = opportunity_id
        self.title = opportunity_data.get('title', '')
        self.description = opportunity_data.get('description', '')
        self.opportunity_type = opportunity_data.get('type', 'market')
        
        # Scoring
        self.market_size = opportunity_data.get('market_size', 0.0)
        self.growth_potential = opportunity_data.get('growth_potential', 0.0)
        self.competitive_advantage = opportunity_data.get('competitive_advantage', 0.0)
        self.implementation_difficulty = opportunity_data.get('implementation_difficulty', 0.5)
        self.resource_requirements = opportunity_data.get('resource_requirements', 0.5)
        
        # Overall score
        self.opportunity_score = self._calculate_opportunity_score()
        
        # Timeline
        self.estimated_timeline = opportunity_data.get('estimated_timeline', 'unknown')
        self.priority_level = opportunity_data.get('priority_level', 'medium')
        
        # Context
        self.market_context = opportunity_data.get('market_context', {})
        self.competitive_context = opportunity_data.get('competitive_context', {})
        
        self.created_at = datetime.utcnow()
    
    def _calculate_opportunity_score(self) -> float:
        """Calculate overall opportunity score"""
        # Weighted scoring model
        score = (
            self.market_size * 0.3 +
            self.growth_potential * 0.25 +
            self.competitive_advantage * 0.25 +
            (1 - self.implementation_difficulty) * 0.1 +
            (1 - self.resource_requirements) * 0.1
        )
        return min(1.0, max(0.0, score))

class RiskAssessment:
    """Strategic risk assessment"""
    def __init__(self, risk_id: str, risk_data: Dict[str, Any]):
        self.id = risk_id
        self.title = risk_data.get('title', '')
        self.description = risk_data.get('description', '')
        self.risk_type = risk_data.get('type', 'market')  # market, competitive, operational, financial
        
        # Risk metrics
        self.probability = risk_data.get('probability', 0.5)  # 0-1
        self.impact_severity = risk_data.get('impact_severity', 0.5)  # 0-1
        self.risk_score = self.probability * self.impact_severity
        
        # Mitigation
        self.mitigation_strategies = risk_data.get('mitigation_strategies', [])
        self.mitigation_cost = risk_data.get('mitigation_cost', 0.0)
        self.mitigation_effectiveness = risk_data.get('mitigation_effectiveness', 0.0)
        
        # Timeline
        self.time_horizon = risk_data.get('time_horizon', 'medium')
        self.early_warning_indicators = risk_data.get('early_warning_indicators', [])
        
        self.created_at = datetime.utcnow()

class StrategicIntelligenceEngine:
    """Core strategic intelligence and analysis engine"""
    
    def __init__(self):
        self.competitive_intelligence: Dict[str, CompetitiveIntelligence] = {}
        self.market_intelligence: Dict[str, MarketIntelligence] = {}
        self.strategic_trends: Dict[str, StrategicTrend] = {}
        self.opportunities: Dict[str, OpportunityAssessment] = {}
        self.risks: Dict[str, RiskAssessment] = {}
        
        # Analytics
        self.market_performance_history: List[Dict[str, Any]] = []
        self.competitive_position_history: List[Dict[str, Any]] = []
        self.strategic_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Intelligence sources
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.intelligence_confidence: Dict[str, float] = {}
        
        # KSE integration for universal intelligence substrate
        self.kse_client = LiftKSEClient()
        self.latest_strategic_insights = []
        self.latest_competitive_analysis = []
        self.latest_market_intelligence = {}
        
    async def analyze_competitive_landscape(self, analysis_data: Dict[str, Any]) -> CompetitiveIntelligence:
        """Analyze competitive landscape and positioning with KSE intelligence enhancement"""
        try:
            competitor_name = analysis_data.get('competitor_name')
            if not competitor_name:
                raise ValueError("competitor_name is required")
            
            # Retrieve strategic intelligence from KSE for competitive analysis enhancement
            strategic_intelligence = await self.retrieve_strategic_intelligence(
                query=f"competitive analysis {competitor_name} market positioning strategic insights",
                domain="strategic"
            )
            
            # Extract performance data with KSE enhancement
            our_performance = analysis_data.get('our_performance', 0.0)
            competitor_performance = analysis_data.get('competitor_performance', 0.0)
            performance_gap = competitor_performance - our_performance
            
            # Market share analysis
            our_market_share = analysis_data.get('our_market_share', 0.0)
            competitor_market_share = analysis_data.get('competitor_market_share', 0.0)
            
            # SWOT analysis enhanced with KSE insights
            strengths = analysis_data.get('strengths', [])
            weaknesses = analysis_data.get('weaknesses', [])
            opportunities = analysis_data.get('opportunities', [])
            threats = analysis_data.get('threats', [])
            
            # Enhance SWOT with KSE strategic intelligence
            kse_opportunities = strategic_intelligence.get('strategic_opportunities', [])
            kse_risks = strategic_intelligence.get('risk_indicators', [])
            
            if kse_opportunities:
                opportunities.extend(kse_opportunities[:3])  # Add top 3 KSE opportunities
            if kse_risks:
                threats.extend(kse_risks[:3])  # Add top 3 KSE risk indicators
            
            # Create competitive intelligence with enhanced confidence
            enhanced_confidence = min(0.95, analysis_data.get('confidence', 0.7) + 0.15)  # Boost confidence with KSE
            
            competitive_intel = CompetitiveIntelligence(
                competitor_name=competitor_name,
                our_performance=our_performance,
                competitor_performance=competitor_performance,
                performance_gap=performance_gap,
                market_share_us=our_market_share,
                market_share_them=competitor_market_share,
                strengths=strengths,
                weaknesses=weaknesses,
                opportunities=opportunities,
                threats=threats,
                source=f"{analysis_data.get('source', 'internal_analysis')}_kse_enhanced",
                confidence=enhanced_confidence
            )
            
            self.competitive_intelligence[competitive_intel.id] = competitive_intel
            
            # Update competitive position history
            position_data = {
                'timestamp': datetime.utcnow(),
                'competitor': competitor_name,
                'performance_gap': performance_gap,
                'market_share_gap': competitor_market_share - our_market_share,
                'kse_enhanced': True,
                'strategic_insights_count': len(strategic_intelligence.get('competitive_insights', []))
            }
            self.competitive_position_history.append(position_data)
            
            # Update latest analysis for enrichment
            self.latest_competitive_analysis = [competitive_intel.dict() if hasattr(competitive_intel, 'dict') else str(competitive_intel)]
            self.latest_strategic_insights = strategic_intelligence.get('competitive_insights', [])
            
            # Enrich intelligence layer with competitive analysis
            enrichment_data = {
                'domain': 'strategic',
                'analysis_type': 'competitive',
                'competitor': competitor_name,
                'competitive_insights': self.latest_competitive_analysis,
                'market_intelligence': [{
                    'market_segment': analysis_data.get('market_segment', 'general'),
                    'performance_gap': performance_gap,
                    'market_share_gap': competitor_market_share - our_market_share,
                    'confidence': enhanced_confidence,
                    'kse_enhancement': {
                        'opportunities_added': len(kse_opportunities),
                        'risks_identified': len(kse_risks),
                        'insights_leveraged': len(strategic_intelligence.get('competitive_insights', []))
                    }
                }]
            }
            
            await self.enrich_strategic_intelligence_layer(
                enrichment_data,
                trace_id=f"competitive_{competitor_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            
            logger.info(f"KSE-enhanced competitive analysis for {competitor_name}: {performance_gap:.2f} performance gap, {enhanced_confidence:.2f} confidence")
            return competitive_intel
            
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {e}")
            raise
    
    async def retrieve_strategic_intelligence(self, query: str, domain: str = "strategic") -> Dict[str, Any]:
        """Retrieve strategic intelligence data from KSE universal substrate"""
        try:
            # Use KSE hybrid search for comprehensive strategic intelligence retrieval
            results = await self.kse_client.hybrid_search(
                query=query,
                domain=domain,
                limit=20,
                include_embeddings=True,
                include_concepts=True,
                include_knowledge_graph=True
            )
            
            # Extract strategic intelligence insights from KSE results
            strategic_intelligence = {
                'competitive_insights': [],
                'market_trends': [],
                'strategic_opportunities': [],
                'risk_indicators': [],
                'recommendations': [],
                'confidence_scores': []
            }
            
            for result in results:
                if 'competitive_insights' in result:
                    strategic_intelligence['competitive_insights'].extend(result['competitive_insights'])
                if 'market_trends' in result:
                    strategic_intelligence['market_trends'].extend(result['market_trends'])
                if 'strategic_opportunities' in result:
                    strategic_intelligence['strategic_opportunities'].extend(result['strategic_opportunities'])
                if 'risk_indicators' in result:
                    strategic_intelligence['risk_indicators'].extend(result['risk_indicators'])
                if 'recommendations' in result:
                    strategic_intelligence['recommendations'].extend(result['recommendations'])
                if 'confidence' in result:
                    strategic_intelligence['confidence_scores'].append(result['confidence'])
            
            self.latest_market_intelligence = strategic_intelligence
            return strategic_intelligence
            
        except Exception as e:
            logger.error(f"Failed to retrieve strategic intelligence: {e}")
            return {}
    
    async def enrich_strategic_intelligence_layer(self, data: Dict[str, Any], trace_id: str = None) -> bool:
        """Write back strategic insights, competitive analysis, and market intelligence to enrich KSE intelligence layer"""
        try:
            import uuid
            
            # Create comprehensive trace for strategic intelligence enrichment
            trace_data = {
                'service': 'strategic-intelligence',
                'timestamp': datetime.utcnow().isoformat(),
                'trace_id': trace_id or str(uuid.uuid4()),
                'operation': 'strategic_intelligence_processing',
                'data': data,
                'competitive_analyses': len(self.competitive_intelligence),
                'market_intelligence_reports': len(self.market_intelligence),
                'strategic_insights': self.latest_strategic_insights,
                'competitive_analysis': self.latest_competitive_analysis
            }
            
            # Store trace in KSE for intelligence layer enrichment
            await self.kse_client.store_trace(trace_data)
            
            # Store competitive insights as entities for future intelligence retrieval
            if 'competitive_insights' in data:
                for insight in data['competitive_insights']:
                    entity_data = {
                        'type': 'competitive_insight',
                        'domain': data.get('domain', 'strategic'),
                        'content': insight,
                        'confidence': insight.get('confidence', 0.8) if isinstance(insight, dict) else 0.8,
                        'source': 'strategic_intelligence',
                        'metadata': {
                            'competitor': insight.get('competitor', 'unknown') if isinstance(insight, dict) else 'unknown',
                            'analysis_type': data.get('analysis_type', 'competitive'),
                            'discovery_time': datetime.utcnow().isoformat(),
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            # Store market intelligence for future strategic analysis
            if 'market_intelligence' in data:
                for intelligence in data['market_intelligence']:
                    entity_data = {
                        'type': 'market_intelligence',
                        'domain': data.get('domain', 'strategic'),
                        'content': intelligence,
                        'confidence': intelligence.get('confidence', 0.8) if isinstance(intelligence, dict) else 0.8,
                        'source': 'strategic_intelligence',
                        'metadata': {
                            'market_segment': intelligence.get('market_segment', 'general') if isinstance(intelligence, dict) else 'general',
                            'intelligence_type': data.get('intelligence_type', 'market'),
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            logger.info(f"Successfully enriched strategic intelligence layer with trace {trace_data['trace_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enrich strategic intelligence layer: {e}")
            return False
    
    async def analyze_market_intelligence(self, market_data: Dict[str, Any]) -> MarketIntelligence:
        """Analyze market trends and intelligence"""
        try:
            market_segment = market_data.get('market_segment')
            if not market_segment:
                raise ValueError("market_segment is required")
            
            # Market metrics
            market_size = market_data.get('market_size', 0.0)
            market_growth_rate = market_data.get('market_growth_rate', 0.0)
            our_market_share = market_data.get('our_market_share', 0.0)
            
            # Trends analysis
            emerging_trends = market_data.get('emerging_trends', [])
            declining_trends = market_data.get('declining_trends', [])
            
            # Opportunities and threats
            market_opportunities = market_data.get('market_opportunities', [])
            threat_indicators = market_data.get('threat_indicators', [])
            
            # Predictions
            predicted_growth = market_data.get('predicted_growth', market_growth_rate)
            confidence_lower = market_data.get('confidence_lower', predicted_growth * 0.8)
            confidence_upper = market_data.get('confidence_upper', predicted_growth * 1.2)
            
            # Create market intelligence
            market_intel = MarketIntelligence(
                market_segment=market_segment,
                market_size=market_size,
                market_growth_rate=market_growth_rate,
                our_market_share=our_market_share,
                emerging_trends=emerging_trends,
                declining_trends=declining_trends,
                market_opportunities=market_opportunities,
                threat_indicators=threat_indicators,
                predicted_growth=predicted_growth,
                confidence_interval=(confidence_lower, confidence_upper),
                data_sources=market_data.get('data_sources', ['internal_analysis'])
            )
            
            self.market_intelligence[market_intel.id] = market_intel
            
            # Update market performance history
            performance_data = {
                'timestamp': datetime.utcnow(),
                'market_segment': market_segment,
                'market_size': market_size,
                'growth_rate': market_growth_rate,
                'our_share': our_market_share
            }
            self.market_performance_history.append(performance_data)
            
            logger.info(f"Analyzed market intelligence for {market_segment}: {market_growth_rate:.1f}% growth, {our_market_share:.1%} share")
            return market_intel
            
        except Exception as e:
            logger.error(f"Error analyzing market intelligence: {e}")
            raise
    
    async def identify_strategic_trends(self, trend_data: Dict[str, Any]) -> StrategicTrend:
        """Identify and analyze strategic trends"""
        try:
            trend_id = trend_data.get('trend_id', f"trend_{datetime.utcnow().timestamp()}")
            
            trend = StrategicTrend(trend_id, trend_data)
            self.strategic_trends[trend_id] = trend
            
            # Add initial data point
            data_point = {
                'timestamp': datetime.utcnow(),
                'value': trend_data.get('initial_value', 0.0),
                'source': trend_data.get('source', 'manual'),
                'confidence': trend_data.get('confidence_score', 0.5)
            }
            trend.data_points.append(data_point)
            
            logger.info(f"Identified strategic trend: {trend.name} ({trend.trend_type})")
            return trend
            
        except Exception as e:
            logger.error(f"Error identifying strategic trend: {e}")
            raise
    
    async def assess_opportunity(self, opportunity_data: Dict[str, Any]) -> OpportunityAssessment:
        """Assess business opportunity"""
        try:
            opportunity_id = opportunity_data.get('opportunity_id', f"opp_{datetime.utcnow().timestamp()}")
            
            opportunity = OpportunityAssessment(opportunity_id, opportunity_data)
            self.opportunities[opportunity_id] = opportunity
            
            logger.info(f"Assessed opportunity: {opportunity.title} (score: {opportunity.opportunity_score:.2f})")
            return opportunity
            
        except Exception as e:
            logger.error(f"Error assessing opportunity: {e}")
            raise
    
    async def assess_risk(self, risk_data: Dict[str, Any]) -> RiskAssessment:
        """Assess strategic risk"""
        try:
            risk_id = risk_data.get('risk_id', f"risk_{datetime.utcnow().timestamp()}")
            
            risk = RiskAssessment(risk_id, risk_data)
            self.risks[risk_id] = risk
            
            logger.info(f"Assessed risk: {risk.title} (score: {risk.risk_score:.2f})")
            return risk
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            raise
    
    async def generate_strategic_recommendations(self) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on intelligence"""
        try:
            recommendations = []
            
            # Opportunity-based recommendations
            top_opportunities = sorted(
                self.opportunities.values(),
                key=lambda x: x.opportunity_score,
                reverse=True
            )[:5]
            
            for opp in top_opportunities:
                if opp.opportunity_score > 0.7:
                    recommendations.append({
                        'type': 'opportunity',
                        'priority': 'high',
                        'title': f"Pursue {opp.title}",
                        'description': f"High-value opportunity with score {opp.opportunity_score:.2f}",
                        'rationale': f"Market size: {opp.market_size}, Growth potential: {opp.growth_potential}",
                        'timeline': opp.estimated_timeline,
                        'confidence': 0.8
                    })
            
            # Risk-based recommendations
            high_risks = [risk for risk in self.risks.values() if risk.risk_score > 0.6]
            for risk in high_risks:
                recommendations.append({
                    'type': 'risk_mitigation',
                    'priority': 'high' if risk.risk_score > 0.8 else 'medium',
                    'title': f"Mitigate {risk.title}",
                    'description': f"High-risk scenario with score {risk.risk_score:.2f}",
                    'rationale': f"Probability: {risk.probability:.1%}, Impact: {risk.impact_severity:.1%}",
                    'mitigation_strategies': risk.mitigation_strategies,
                    'confidence': 0.7
                })
            
            # Competitive recommendations
            competitive_gaps = []
            for intel in self.competitive_intelligence.values():
                if intel.performance_gap > 0.1:  # Competitor performing 10% better
                    competitive_gaps.append(intel)
            
            for gap in competitive_gaps:
                recommendations.append({
                    'type': 'competitive',
                    'priority': 'medium',
                    'title': f"Close gap with {gap.competitor_name}",
                    'description': f"Performance gap of {gap.performance_gap:.1%}",
                    'rationale': f"Competitor strengths: {', '.join(gap.strengths[:3])}",
                    'focus_areas': gap.weaknesses,
                    'confidence': gap.confidence
                })
            
            # Market trend recommendations
            emerging_trends = []
            for intel in self.market_intelligence.values():
                emerging_trends.extend(intel.emerging_trends)
            
            if emerging_trends:
                recommendations.append({
                    'type': 'market_trend',
                    'priority': 'medium',
                    'title': "Capitalize on emerging trends",
                    'description': f"Monitor and adapt to {len(emerging_trends)} emerging trends",
                    'trends': emerging_trends,
                    'confidence': 0.6
                })
            
            # Sort by priority and confidence
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(
                key=lambda x: (priority_order.get(x['priority'], 0), x.get('confidence', 0)),
                reverse=True
            )
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            raise
    
    async def get_strategic_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive strategic intelligence dashboard"""
        try:
            # Market position summary
            market_position = {}
            if self.market_intelligence:
                total_market_size = sum(m.market_size for m in self.market_intelligence.values())
                avg_growth_rate = statistics.mean(m.market_growth_rate for m in self.market_intelligence.values())
                avg_market_share = statistics.mean(m.our_market_share for m in self.market_intelligence.values())
                
                market_position = {
                    'total_addressable_market': total_market_size,
                    'average_growth_rate': avg_growth_rate,
                    'average_market_share': avg_market_share,
                    'market_segments': len(self.market_intelligence)
                }
            
            # Competitive position
            competitive_position = {}
            if self.competitive_intelligence:
                avg_performance_gap = statistics.mean(c.performance_gap for c in self.competitive_intelligence.values())
                competitors_ahead = len([c for c in self.competitive_intelligence.values() if c.performance_gap > 0])
                
                competitive_position = {
                    'average_performance_gap': avg_performance_gap,
                    'competitors_monitored': len(self.competitive_intelligence),
                    'competitors_ahead': competitors_ahead,
                    'competitive_advantage': avg_performance_gap < 0
                }
            
            # Opportunity pipeline
            opportunity_pipeline = {}
            if self.opportunities:
                high_value_opportunities = len([o for o in self.opportunities.values() if o.opportunity_score > 0.7])
                avg_opportunity_score = statistics.mean(o.opportunity_score for o in self.opportunities.values())
                
                opportunity_pipeline = {
                    'total_opportunities': len(self.opportunities),
                    'high_value_opportunities': high_value_opportunities,
                    'average_opportunity_score': avg_opportunity_score
                }
            
            # Risk profile
            risk_profile = {}
            if self.risks:
                high_risks = len([r for r in self.risks.values() if r.risk_score > 0.7])
                avg_risk_score = statistics.mean(r.risk_score for r in self.risks.values())
                
                risk_profile = {
                    'total_risks': len(self.risks),
                    'high_risks': high_risks,
                    'average_risk_score': avg_risk_score
                }
            
            # Strategic recommendations
            recommendations = await self.generate_strategic_recommendations()
            
            dashboard = {
                'timestamp': datetime.utcnow().isoformat(),
                'market_position': market_position,
                'competitive_position': competitive_position,
                'opportunity_pipeline': opportunity_pipeline,
                'risk_profile': risk_profile,
                'strategic_trends': len(self.strategic_trends),
                'top_recommendations': recommendations[:5],
                'intelligence_confidence': statistics.mean(self.intelligence_confidence.values()) if self.intelligence_confidence else 0.5
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating strategic dashboard: {e}")
            raise
    
    async def update_trend_data(self, trend_id: str, data_point: Dict[str, Any]):
        """Update trend with new data point"""
        try:
            if trend_id not in self.strategic_trends:
                raise ValueError(f"Trend {trend_id} not found")
            
            trend = self.strategic_trends[trend_id]
            
            new_data_point = {
                'timestamp': datetime.utcnow(),
                'value': data_point.get('value', 0.0),
                'source': data_point.get('source', 'update'),
                'confidence': data_point.get('confidence', 0.5)
            }
            
            trend.data_points.append(new_data_point)
            trend.last_updated = datetime.utcnow()
            
            # Update confidence based on data quality and recency
            recent_points = [dp for dp in trend.data_points if 
                           (datetime.utcnow() - dp['timestamp']).days <= 30]
            
            if recent_points:
                trend.confidence_score = statistics.mean(dp['confidence'] for dp in recent_points)
            
            logger.info(f"Updated trend {trend.name} with new data point: {new_data_point['value']}")
            
        except Exception as e:
            logger.error(f"Error updating trend data: {e}")
            raise

# Initialize the strategic intelligence engine
strategic_engine = StrategicIntelligenceEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Strategic Intelligence Service...")
    
    # Initialize sample strategic trends
    await strategic_engine.identify_strategic_trends({
        'trend_id': 'ai_adoption',
        'name': 'AI Adoption in Business',
        'description': 'Increasing adoption of AI technologies across industries',
        'trend_type': 'technology',
        'impact_level': 'high',
        'time_horizon': 'medium',
        'confidence_score': 0.8,
        'initial_value': 0.3
    })
    
    await strategic_engine.identify_strategic_trends({
        'trend_id': 'data_privacy_regulation',
        'name': 'Data Privacy Regulation',
        'description': 'Increasing regulatory focus on data privacy and protection',
        'trend_type': 'regulatory',
        'impact_level': 'high',
        'time_horizon': 'short',
        'confidence_score': 0.9,
        'initial_value': 0.7
    })
    
    logger.info("Strategic Intelligence Service started successfully")
    yield
    logger.info("Strategic Intelligence Service shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LiftOS Strategic Intelligence Service",
    description="Market intelligence, competitive analysis, and strategic insights",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "strategic-intelligence",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/competitive/analyze")
async def analyze_competitive(data: Dict[str, Any]):
    """Analyze competitive landscape"""
    try:
        intel = await strategic_engine.analyze_competitive_landscape(data)
        return {"status": "success", "intelligence": intel.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing competitive landscape: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market/analyze")
async def analyze_market(data: Dict[str, Any]):
    """Analyze market intelligence"""
    try:
        intel = await strategic_engine.analyze_market_intelligence(data)
        return {"status": "success", "intelligence": intel.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing market intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trends/identify")
async def identify_trend(data: Dict[str, Any]):
    """Identify strategic trend"""
    try:
        trend = await strategic_engine.identify_strategic_trends(data)
        return {"status": "success", "trend": {
            "id": trend.id,
            "name": trend.name,
            "type": trend.trend_type,
            "confidence": trend.confidence_score
        }}
    except Exception as e:
        logger.error(f"Error identifying trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trends/{trend_id}/update")
async def update_trend(trend_id: str, data: Dict[str, Any]):
    """Update trend with new data"""
    try:
        await strategic_engine.update_trend_data(trend_id, data)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/opportunities/assess")
async def assess_opportunity(data: Dict[str, Any]):
    """Assess business opportunity"""
    try:
        opportunity = await strategic_engine.assess_opportunity(data)
        return {"status": "success", "opportunity": {
            "id": opportunity.id,
            "title": opportunity.title,
            "score": opportunity.opportunity_score,
            "priority": opportunity.priority_level
        }}
    except Exception as e:
        logger.error(f"Error assessing opportunity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risks/assess")
async def assess_risk(data: Dict[str, Any]):
    """Assess strategic risk"""
    try:
        risk = await strategic_engine.assess_risk(data)
        return {"status": "success", "risk": {
            "id": risk.id,
            "title": risk.title,
            "score": risk.risk_score,
            "probability": risk.probability,
            "impact": risk.impact_severity
        }}
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations")
async def get_recommendations():
    """Get strategic recommendations"""
    try:
        recommendations = await strategic_engine.generate_strategic_recommendations()
        return {"status": "success", "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_strategic_dashboard():
    """Get strategic intelligence dashboard"""
    try:
        dashboard = await strategic_engine.get_strategic_dashboard()
        return {"status": "success", "dashboard": dashboard}
    except Exception as e:
        logger.error(f"Error getting strategic dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/competitive")
async def get_competitive_intelligence():
    """Get competitive intelligence summary"""
    try:
        intel_list = [intel.dict() for intel in strategic_engine.competitive_intelligence.values()]
        return {"status": "success", "competitive_intelligence": intel_list}
    except Exception as e:
        logger.error(f"Error getting competitive intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market")
async def get_market_intelligence():
    """Get market intelligence summary"""
    try:
        intel_list = [intel.dict() for intel in strategic_engine.market_intelligence.values()]
        return {"status": "success", "market_intelligence": intel_list}
    except Exception as e:
        logger.error(f"Error getting market intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8015)