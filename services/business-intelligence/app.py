"""
LiftOS Business Intelligence Dashboard Service
Comprehensive business observability visualization and reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import json
import aiohttp
from collections import defaultdict
import statistics

# Import models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.business import (
    BusinessDashboard, BusinessReport, BusinessAlert, BusinessMetric,
    MetricType, MetricFrequency
)
from shared.kse_sdk.client import LiftKSEClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardWidget:
    """Dashboard widget configuration"""
    def __init__(self, widget_id: str, widget_config: Dict[str, Any]):
        self.id = widget_id
        self.title = widget_config.get('title', '')
        self.widget_type = widget_config.get('type', 'metric')  # metric, chart, table, kpi
        self.data_source = widget_config.get('data_source', '')
        self.query = widget_config.get('query', {})
        self.visualization_config = widget_config.get('visualization', {})
        self.refresh_interval = widget_config.get('refresh_interval', 300)  # seconds
        self.position = widget_config.get('position', {'x': 0, 'y': 0, 'width': 4, 'height': 3})
        self.filters = widget_config.get('filters', {})
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()

class ReportTemplate:
    """Report template configuration"""
    def __init__(self, template_id: str, template_config: Dict[str, Any]):
        self.id = template_id
        self.name = template_config.get('name', '')
        self.description = template_config.get('description', '')
        self.report_type = template_config.get('type', 'performance')
        self.sections = template_config.get('sections', [])
        self.data_sources = template_config.get('data_sources', [])
        self.schedule = template_config.get('schedule', None)
        self.recipients = template_config.get('recipients', [])
        self.format_options = template_config.get('format_options', {})
        self.created_at = datetime.utcnow()

class BusinessIntelligenceEngine:
    """Core business intelligence and dashboard engine"""
    
    def __init__(self):
        self.dashboards: Dict[str, BusinessDashboard] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.generated_reports: List[BusinessReport] = []
        self.alerts: Dict[str, BusinessAlert] = {}
        
        # Data cache
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Service endpoints
        self.service_endpoints = {
            'business-metrics': 'http://localhost:8012',
            'user-analytics': 'http://localhost:8013',
            'impact-monitoring': 'http://localhost:8014',
            'strategic-intelligence': 'http://localhost:8015'
        }
        
        # Real-time data
        self.real_time_metrics: Dict[str, Any] = {}
        self.metric_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # KSE integration for universal intelligence substrate
        self.kse_client = LiftKSEClient()
        self.latest_insights = []
        self.latest_patterns = []
        self.latest_business_intelligence = {}
        
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> BusinessDashboard:
        """Create a new business dashboard"""
        try:
            dashboard = BusinessDashboard(
                name=dashboard_config['name'],
                description=dashboard_config.get('description', ''),
                dashboard_type=dashboard_config.get('type', 'operational'),
                kpi_ids=dashboard_config.get('kpi_ids', []),
                metric_ids=dashboard_config.get('metric_ids', []),
                chart_configs=dashboard_config.get('chart_configs', []),
                owner=dashboard_config.get('owner', 'system'),
                viewers=dashboard_config.get('viewers', []),
                refresh_frequency=MetricFrequency(dashboard_config.get('refresh_frequency', 'hourly')),
                auto_refresh=dashboard_config.get('auto_refresh', True),
                layout_config=dashboard_config.get('layout_config', {})
            )
            
            self.dashboards[dashboard.id] = dashboard
            
            logger.info(f"Created dashboard: {dashboard.name} ({dashboard.dashboard_type})")
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    async def add_widget_to_dashboard(self, dashboard_id: str, widget_config: Dict[str, Any]) -> DashboardWidget:
        """Add a widget to a dashboard"""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            widget_id = widget_config.get('widget_id', f"widget_{datetime.utcnow().timestamp()}")
            widget = DashboardWidget(widget_id, widget_config)
            
            self.widgets[widget_id] = widget
            
            # Add widget to dashboard layout
            dashboard = self.dashboards[dashboard_id]
            if 'widgets' not in dashboard.layout_config:
                dashboard.layout_config['widgets'] = []
            
            dashboard.layout_config['widgets'].append({
                'widget_id': widget_id,
                'position': widget.position
            })
            
            logger.info(f"Added widget {widget.title} to dashboard {dashboard.name}")
            return widget
            
        except Exception as e:
            logger.error(f"Error adding widget to dashboard: {e}")
            raise
    
    async def fetch_service_data(self, service_name: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch data from a LiftOS service"""
        try:
            if service_name not in self.service_endpoints:
                raise ValueError(f"Unknown service: {service_name}")
            
            base_url = self.service_endpoints[service_name]
            url = f"{base_url}/{endpoint.lstrip('/')}"
            
            # Check cache first
            cache_key = f"{service_name}_{endpoint}_{str(params)}"
            if cache_key in self.data_cache:
                cache_time = self.cache_timestamps.get(cache_key, datetime.min)
                if (datetime.utcnow() - cache_time).seconds < 300:  # 5 minute cache
                    return self.data_cache[cache_key]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache the data
                        self.data_cache[cache_key] = data
                        self.cache_timestamps[cache_key] = datetime.utcnow()
                        
                        return data
                    else:
                        logger.warning(f"Service {service_name} returned status {response.status}")
                        return {"error": f"Service unavailable: {response.status}"}
            
        except Exception as e:
            logger.error(f"Error fetching data from {service_name}: {e}")
            return {"error": str(e)}
    
    async def retrieve_business_intelligence(self, query: str, domain: str = "business") -> Dict[str, Any]:
        """Retrieve business intelligence data from KSE universal substrate"""
        try:
            # Use KSE hybrid search for comprehensive business intelligence retrieval
            results = await self.kse_client.hybrid_search(
                query=query,
                domain=domain,
                limit=15,
                include_embeddings=True,
                include_concepts=True,
                include_knowledge_graph=True
            )
            
            # Extract business intelligence insights from KSE results
            business_intelligence = {
                'metrics': [],
                'patterns': [],
                'trends': [],
                'insights': [],
                'recommendations': [],
                'confidence_scores': []
            }
            
            for result in results:
                if 'metrics' in result:
                    business_intelligence['metrics'].extend(result['metrics'])
                if 'patterns' in result:
                    business_intelligence['patterns'].extend(result['patterns'])
                if 'trends' in result:
                    business_intelligence['trends'].extend(result['trends'])
                if 'insights' in result:
                    business_intelligence['insights'].extend(result['insights'])
                if 'recommendations' in result:
                    business_intelligence['recommendations'].extend(result['recommendations'])
                if 'confidence' in result:
                    business_intelligence['confidence_scores'].append(result['confidence'])
            
            self.latest_business_intelligence = business_intelligence
            return business_intelligence
            
        except Exception as e:
            logger.error(f"Failed to retrieve business intelligence: {e}")
            return {}
    
    async def enrich_business_intelligence_layer(self, data: Dict[str, Any], trace_id: str = None) -> bool:
        """Write back business insights, dashboards, and analytics to enrich KSE intelligence layer"""
        try:
            import uuid
            
            # Create comprehensive trace for business intelligence enrichment
            trace_data = {
                'service': 'business-intelligence',
                'timestamp': datetime.utcnow().isoformat(),
                'trace_id': trace_id or str(uuid.uuid4()),
                'operation': 'business_intelligence_processing',
                'data': data,
                'dashboards_created': len(self.dashboards),
                'widgets_active': len(self.widgets),
                'insights_generated': self.latest_insights,
                'patterns_discovered': self.latest_patterns
            }
            
            # Store trace in KSE for intelligence layer enrichment
            await self.kse_client.store_trace(trace_data)
            
            # Store business insights as entities for future intelligence retrieval
            if 'insights' in data:
                for insight in data['insights']:
                    entity_data = {
                        'type': 'business_insight',
                        'domain': data.get('domain', 'business'),
                        'content': insight,
                        'confidence': insight.get('confidence', 0.7) if isinstance(insight, dict) else 0.7,
                        'source': 'business_intelligence',
                        'metadata': {
                            'dashboard_context': data.get('dashboard_id', 'unknown'),
                            'metric_type': data.get('metric_type', 'general'),
                            'discovery_time': datetime.utcnow().isoformat(),
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            # Store dashboard analytics for future business intelligence
            if 'dashboard_analytics' in data:
                for analytics in data['dashboard_analytics']:
                    entity_data = {
                        'type': 'dashboard_analytics',
                        'domain': data.get('domain', 'business'),
                        'content': analytics,
                        'confidence': analytics.get('confidence', 0.8) if isinstance(analytics, dict) else 0.8,
                        'source': 'business_intelligence',
                        'metadata': {
                            'dashboard_type': analytics.get('type', 'operational') if isinstance(analytics, dict) else 'operational',
                            'performance_metrics': analytics.get('metrics', {}) if isinstance(analytics, dict) else {},
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            logger.info(f"Successfully enriched business intelligence layer with trace {trace_data['trace_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enrich business intelligence layer: {e}")
            return False
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get complete dashboard data with all widgets enhanced by KSE intelligence"""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
            
            dashboard = self.dashboards[dashboard_id]
            
            # Retrieve business intelligence from KSE for dashboard enhancement
            business_intelligence = await self.retrieve_business_intelligence(
                query=f"dashboard {dashboard.name} {dashboard.dashboard_type} business metrics",
                domain="business"
            )
            
            dashboard_data = {
                "dashboard": dashboard.dict(),
                "widgets": {},
                "last_updated": datetime.utcnow().isoformat(),
                "kse_intelligence": {
                    "insights": business_intelligence.get('insights', []),
                    "patterns": business_intelligence.get('patterns', []),
                    "recommendations": business_intelligence.get('recommendations', [])
                }
            }
            
            # Get data for each widget with KSE enhancement
            widget_ids = [w['widget_id'] for w in dashboard.layout_config.get('widgets', [])]
            
            for widget_id in widget_ids:
                if widget_id in self.widgets:
                    widget = self.widgets[widget_id]
                    widget_data = await self._get_widget_data(widget)
                    
                    # Enhance widget data with KSE intelligence
                    enhanced_widget_data = {
                        'config': {
                            'id': widget.id,
                            'title': widget.title,
                            'type': widget.widget_type,
                            'position': widget.position,
                            'visualization': widget.visualization_config
                        },
                        'data': widget_data,
                        'kse_insights': business_intelligence.get('insights', [])[:3],  # Top 3 insights per widget
                        'confidence': business_intelligence.get('confidence_scores', [0.8])[0] if business_intelligence.get('confidence_scores') else 0.8
                    }
                    
                    dashboard_data['widgets'][widget_id] = enhanced_widget_data
            
            # Update latest insights and patterns for enrichment
            self.latest_insights = business_intelligence.get('insights', [])
            self.latest_patterns = business_intelligence.get('patterns', [])
            
            # Enrich intelligence layer with dashboard analytics
            enrichment_data = {
                'domain': 'business',
                'dashboard_id': dashboard_id,
                'dashboard_analytics': [{
                    'type': dashboard.dashboard_type,
                    'metrics': list(dashboard_data['widgets'].keys()),
                    'confidence': 0.85,
                    'performance': {
                        'widgets_count': len(dashboard_data['widgets']),
                        'intelligence_enhanced': True,
                        'kse_insights_count': len(business_intelligence.get('insights', []))
                    }
                }],
                'insights': self.latest_insights
            }
            
            await self.enrich_business_intelligence_layer(
                enrichment_data,
                trace_id=f"dashboard_{dashboard_id}"
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        try:
            data_source = widget.data_source
            query = widget.query
            
            if data_source == 'business-metrics':
                if widget.widget_type == 'kpi':
                    return await self.fetch_service_data('business-metrics', 'kpis')
                elif widget.widget_type == 'metric':
                    metric_type = query.get('metric_type', 'revenue')
                    return await self.fetch_service_data('business-metrics', f'metrics/{metric_type}')
                else:
                    return await self.fetch_service_data('business-metrics', 'summary')
            
            elif data_source == 'user-analytics':
                if widget.widget_type == 'chart':
                    chart_type = query.get('chart_type', 'users')
                    return await self.fetch_service_data('user-analytics', f'analytics/{chart_type}')
                else:
                    return await self.fetch_service_data('user-analytics', 'analytics/users')
            
            elif data_source == 'impact-monitoring':
                if widget.widget_type == 'table':
                    return await self.fetch_service_data('impact-monitoring', 'pending')
                else:
                    return await self.fetch_service_data('impact-monitoring', 'summary')
            
            elif data_source == 'strategic-intelligence':
                if widget.widget_type == 'recommendations':
                    return await self.fetch_service_data('strategic-intelligence', 'recommendations')
                else:
                    return await self.fetch_service_data('strategic-intelligence', 'dashboard')
            
            else:
                return {"error": f"Unknown data source: {data_source}"}
            
        except Exception as e:
            logger.error(f"Error getting widget data: {e}")
            return {"error": str(e)}
    
    async def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary dashboard"""
        try:
            # Fetch key metrics from all services
            business_summary = await self.fetch_service_data('business-metrics', 'summary')
            user_analytics = await self.fetch_service_data('user-analytics', 'analytics/users')
            impact_summary = await self.fetch_service_data('impact-monitoring', 'summary')
            strategic_dashboard = await self.fetch_service_data('strategic-intelligence', 'dashboard')
            
            # Calculate overall health score
            health_scores = []
            
            if 'summary' in business_summary:
                health_scores.append(business_summary['summary'].get('business_health_score', 0.5))
            
            if 'analytics' in user_analytics:
                satisfaction = user_analytics['analytics'].get('average_satisfaction', 5.0)
                health_scores.append(satisfaction / 10.0)  # Normalize to 0-1
            
            if 'summary' in impact_summary:
                accuracy = impact_summary['summary'].get('prediction_accuracy', 0.5)
                health_scores.append(accuracy)
            
            overall_health = statistics.mean(health_scores) if health_scores else 0.5
            
            # Key performance indicators
            kpis = {
                'overall_health_score': overall_health,
                'business_metrics': business_summary.get('summary', {}),
                'user_engagement': user_analytics.get('analytics', {}),
                'decision_impact': impact_summary.get('summary', {}),
                'strategic_position': strategic_dashboard.get('dashboard', {})
            }
            
            # Trends and insights
            trends = []
            if 'dashboard' in strategic_dashboard:
                strategic_data = strategic_dashboard['dashboard']
                if strategic_data.get('top_recommendations'):
                    trends.extend(strategic_data['top_recommendations'][:3])
            
            # Alerts and notifications
            alerts = []
            if overall_health < 0.6:
                alerts.append({
                    'type': 'warning',
                    'message': 'Overall business health below target',
                    'severity': 'medium'
                })
            
            if 'summary' in impact_summary:
                accuracy = impact_summary['summary'].get('prediction_accuracy', 1.0)
                if accuracy < 0.7:
                    alerts.append({
                        'type': 'warning',
                        'message': 'Decision prediction accuracy below target',
                        'severity': 'medium'
                    })
            
            executive_summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_health_score': overall_health,
                'key_performance_indicators': kpis,
                'strategic_trends': trends,
                'alerts': alerts,
                'data_freshness': {
                    'business_metrics': business_summary.get('status') == 'success',
                    'user_analytics': user_analytics.get('status') == 'success',
                    'impact_monitoring': impact_summary.get('status') == 'success',
                    'strategic_intelligence': strategic_dashboard.get('status') == 'success'
                }
            }
            
            return executive_summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            raise
    
    async def generate_operational_dashboard(self) -> Dict[str, Any]:
        """Generate operational dashboard for day-to-day monitoring"""
        try:
            # Real-time operational metrics
            operational_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_status': {},
                'user_activity': {},
                'decision_pipeline': {},
                'performance_metrics': {}
            }
            
            # System status
            business_health = await self.fetch_service_data('business-metrics', 'health')
            user_health = await self.fetch_service_data('user-analytics', 'health')
            impact_health = await self.fetch_service_data('impact-monitoring', 'health')
            strategic_health = await self.fetch_service_data('strategic-intelligence', 'health')
            
            operational_data['system_status'] = {
                'business_metrics_service': business_health.get('status') == 'healthy',
                'user_analytics_service': user_health.get('status') == 'healthy',
                'impact_monitoring_service': impact_health.get('status') == 'healthy',
                'strategic_intelligence_service': strategic_health.get('status') == 'healthy'
            }
            
            # User activity
            user_analytics = await self.fetch_service_data('user-analytics', 'analytics/users')
            if 'analytics' in user_analytics:
                operational_data['user_activity'] = user_analytics['analytics']
            
            # Decision pipeline
            pending_measurements = await self.fetch_service_data('impact-monitoring', 'pending')
            decision_analytics = await self.fetch_service_data('user-analytics', 'analytics/decisions')
            
            operational_data['decision_pipeline'] = {
                'pending_measurements': pending_measurements.get('pending', []),
                'decision_analytics': decision_analytics.get('analytics', {})
            }
            
            # Performance metrics
            business_summary = await self.fetch_service_data('business-metrics', 'summary')
            if 'summary' in business_summary:
                operational_data['performance_metrics'] = business_summary['summary']
            
            return operational_data
            
        except Exception as e:
            logger.error(f"Error generating operational dashboard: {e}")
            raise
    
    async def create_report_template(self, template_config: Dict[str, Any]) -> ReportTemplate:
        """Create a new report template"""
        try:
            template_id = template_config.get('template_id', f"template_{datetime.utcnow().timestamp()}")
            template = ReportTemplate(template_id, template_config)
            
            self.report_templates[template_id] = template
            
            logger.info(f"Created report template: {template.name}")
            return template
            
        except Exception as e:
            logger.error(f"Error creating report template: {e}")
            raise
    
    async def generate_report(self, template_id: str, report_params: Dict[str, Any] = None) -> BusinessReport:
        """Generate a business report from template"""
        try:
            if template_id not in self.report_templates:
                raise ValueError(f"Report template {template_id} not found")
            
            template = self.report_templates[template_id]
            
            # Determine report period
            period_end = datetime.utcnow()
            period_start = period_end - timedelta(days=report_params.get('period_days', 30))
            
            # Collect data for report sections
            report_data = {
                'metrics': [],
                'kpis': [],
                'insights': [],
                'recommendations': []
            }
            
            # Generate insights based on template type
            if template.report_type == 'performance':
                business_summary = await self.fetch_service_data('business-metrics', 'summary')
                if 'summary' in business_summary:
                    report_data['insights'].append(f"Business health score: {business_summary['summary'].get('business_health_score', 0):.1%}")
            
            elif template.report_type == 'roi':
                roi_analytics = await self.fetch_service_data('impact-monitoring', 'analytics/roi')
                if 'roi_impact' in roi_analytics:
                    roi_data = roi_analytics['roi_impact']
                    report_data['insights'].append(f"ROI: {roi_data.get('roi_percentage', 0):.1f}%")
            
            elif template.report_type == 'strategic':
                recommendations = await self.fetch_service_data('strategic-intelligence', 'recommendations')
                if 'recommendations' in recommendations:
                    report_data['recommendations'] = recommendations['recommendations'][:5]
            
            # Create report
            report = BusinessReport(
                name=template.name,
                description=template.description,
                report_type=template.report_type,
                metrics=report_data['metrics'],
                kpis=report_data['kpis'],
                insights=report_data['insights'],
                recommendations=[r.get('title', '') for r in report_data['recommendations']],
                period_start=period_start,
                period_end=period_end,
                recipients=template.recipients,
                format_type=report_params.get('format', 'json'),
                template_id=template_id
            )
            
            self.generated_reports.append(report)
            
            logger.info(f"Generated report: {report.name} ({report.report_type})")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time business metrics"""
        try:
            # Fetch latest data from all services
            current_time = datetime.utcnow()
            
            real_time_data = {
                'timestamp': current_time.isoformat(),
                'business_metrics': await self.fetch_service_data('business-metrics', 'summary'),
                'user_activity': await self.fetch_service_data('user-analytics', 'analytics/users'),
                'decision_impact': await self.fetch_service_data('impact-monitoring', 'summary'),
                'strategic_insights': await self.fetch_service_data('strategic-intelligence', 'dashboard')
            }
            
            # Store in history for trending
            self.metric_history['real_time_snapshot'].append({
                'timestamp': current_time,
                'data': real_time_data
            })
            
            # Keep only last 100 snapshots
            if len(self.metric_history['real_time_snapshot']) > 100:
                self.metric_history['real_time_snapshot'] = self.metric_history['real_time_snapshot'][-100:]
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            raise

# Initialize the business intelligence engine
bi_engine = BusinessIntelligenceEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Business Intelligence Service...")
    
    # Create default dashboards
    await bi_engine.create_dashboard({
        'name': 'Executive Dashboard',
        'description': 'High-level business performance overview',
        'type': 'executive',
        'owner': 'executive_team',
        'refresh_frequency': 'hourly',
        'layout_config': {
            'theme': 'executive',
            'widgets': []
        }
    })
    
    await bi_engine.create_dashboard({
        'name': 'Operational Dashboard',
        'description': 'Day-to-day operational monitoring',
        'type': 'operational',
        'owner': 'operations_team',
        'refresh_frequency': 'real_time',
        'layout_config': {
            'theme': 'operational',
            'widgets': []
        }
    })
    
    # Create default report templates
    await bi_engine.create_report_template({
        'template_id': 'weekly_performance',
        'name': 'Weekly Performance Report',
        'description': 'Weekly business performance summary',
        'type': 'performance',
        'sections': ['metrics', 'kpis', 'insights'],
        'schedule': 'weekly',
        'recipients': ['management@company.com']
    })
    
    logger.info("Business Intelligence Service started successfully")
    yield
    logger.info("Business Intelligence Service shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LiftOS Business Intelligence Service",
    description="Comprehensive business observability visualization and reporting",
    version="1.0.0",
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
        "service": "business-intelligence",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/dashboards")
async def create_dashboard(data: Dict[str, Any]):
    """Create a new dashboard"""
    try:
        dashboard = await bi_engine.create_dashboard(data)
        return {"status": "success", "dashboard": dashboard.dict()}
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboards")
async def get_dashboards():
    """Get all dashboards"""
    try:
        dashboards = [dashboard.dict() for dashboard in bi_engine.dashboards.values()]
        return {"status": "success", "dashboards": dashboards}
    except Exception as e:
        logger.error(f"Error getting dashboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """Get dashboard with data"""
    try:
        dashboard_data = await bi_engine.get_dashboard_data(dashboard_id)
        return {"status": "success", "dashboard": dashboard_data}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dashboards/{dashboard_id}/widgets")
async def add_widget(dashboard_id: str, data: Dict[str, Any]):
    """Add widget to dashboard"""
    try:
        widget = await bi_engine.add_widget_to_dashboard(dashboard_id, data)
        return {"status": "success", "widget_id": widget.id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding widget: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/executive-summary")
async def get_executive_summary():
    """Get executive summary dashboard"""
    try:
        summary = await bi_engine.generate_executive_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"Error getting executive summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/operational-dashboard")
async def get_operational_dashboard():
    """Get operational dashboard"""
    try:
        dashboard = await bi_engine.generate_operational_dashboard()
        return {"status": "success", "dashboard": dashboard}
    except Exception as e:
        logger.error(f"Error getting operational dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/real-time-metrics")
async def get_real_time_metrics():
    """Get real-time business metrics"""
    try:
        metrics = await bi_engine.get_real_time_metrics()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/templates")
async def create_report_template(data: Dict[str, Any]):
    """Create report template"""
    try:
        template = await bi_engine.create_report_template(data)
        return {"status": "success", "template_id": template.id}
    except Exception as e:
        logger.error(f"Error creating report template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate/{template_id}")
async def generate_report(template_id: str, data: Dict[str, Any] = None):
    """Generate report from template"""
    try:
        report = await bi_engine.generate_report(template_id, data or {})
        return {"status": "success", "report": report.dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports")
async def get_reports():
    """Get generated reports"""
    try:
        reports = [report.dict() for report in bi_engine.generated_reports]
        return {"status": "success", "reports": reports}
    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8016)