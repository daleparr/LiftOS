"""
Intelligent Data Aggregation Engine
Automatically reconciles and aggregates marketing data from multiple sources
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

from shared.models.platform_connections import PlatformConnection, DataQualityMetrics
from shared.services.calendar_consolidation import CalendarConsolidationService
from shared.utils.config import get_service_config
from shared.utils.database import get_database_session
from shared.kse_sdk.client import KSEClient

logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    """Data aggregation strategies"""
    SUM = "sum"  # Sum all values
    AVERAGE = "average"  # Average all values
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by data quality
    LATEST_WINS = "latest_wins"  # Most recent data wins
    HIGHEST_QUALITY = "highest_quality"  # Highest quality source wins
    INTELLIGENT_MERGE = "intelligent_merge"  # AI-powered intelligent merging

class ConflictType(Enum):
    """Types of data conflicts"""
    VALUE_MISMATCH = "value_mismatch"  # Different values for same metric
    MISSING_DATA = "missing_data"  # Data missing from some sources
    DUPLICATE_DATA = "duplicate_data"  # Same data from multiple sources
    TEMPORAL_MISMATCH = "temporal_mismatch"  # Time period misalignment
    ATTRIBUTION_CONFLICT = "attribution_conflict"  # Attribution model differences

@dataclass
class DataSource:
    """Represents a marketing data source"""
    platform_id: str
    platform_name: str
    connection_id: str
    data_quality_score: float
    last_sync: datetime
    attribution_model: str
    time_zone: str
    currency: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataConflict:
    """Represents a data conflict between sources"""
    conflict_type: ConflictType
    metric_name: str
    date: date
    sources_involved: List[str]
    values: Dict[str, Any]
    confidence_scores: Dict[str, float]
    resolution_strategy: str
    resolved_value: Any
    manual_review_required: bool = False

@dataclass
class AggregationResult:
    """Result of data aggregation process"""
    success: bool
    aggregated_data: List[Dict[str, Any]]
    conflicts_detected: List[DataConflict]
    conflicts_resolved: int
    data_quality_score: float
    sources_processed: int
    processing_time: float
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class IntelligentDataAggregationEngine:
    """Engine for intelligent marketing data aggregation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_service_config()
        self.db_session = get_database_session()
        self.kse_client = KSEClient()
        self.calendar_service = CalendarConsolidationService()
        
        # Default aggregation strategies by metric type
        self.metric_strategies = {
            'impressions': AggregationStrategy.SUM,
            'clicks': AggregationStrategy.SUM,
            'conversions': AggregationStrategy.SUM,
            'spend': AggregationStrategy.SUM,
            'revenue': AggregationStrategy.SUM,
            'ctr': AggregationStrategy.WEIGHTED_AVERAGE,
            'cpc': AggregationStrategy.WEIGHTED_AVERAGE,
            'cpm': AggregationStrategy.WEIGHTED_AVERAGE,
            'roas': AggregationStrategy.WEIGHTED_AVERAGE,
            'conversion_rate': AggregationStrategy.WEIGHTED_AVERAGE,
            'frequency': AggregationStrategy.WEIGHTED_AVERAGE,
            'reach': AggregationStrategy.INTELLIGENT_MERGE  # Avoid double counting
        }
        
        # Data quality weights
        self.quality_weights = {
            'platform_reliability': 0.3,
            'data_freshness': 0.2,
            'completeness': 0.2,
            'consistency': 0.15,
            'attribution_accuracy': 0.15
        }
    
    async def aggregate_marketing_data(
        self,
        date_range: Tuple[date, date],
        platforms: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        aggregation_level: str = 'daily'
    ) -> AggregationResult:
        """
        Aggregate marketing data from multiple platforms
        
        Args:
            date_range: Date range to aggregate (start_date, end_date)
            platforms: List of platform IDs to include (None for all)
            metrics: List of metrics to aggregate (None for all)
            aggregation_level: Aggregation level ('daily', 'weekly', 'monthly')
            
        Returns:
            AggregationResult with aggregated data and conflict resolution
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data aggregation for {date_range[0]} to {date_range[1]}")
            
            # Collect data from all sources
            data_sources = await self._collect_data_sources(date_range, platforms)
            
            if not data_sources:
                return AggregationResult(
                    success=False,
                    aggregated_data=[],
                    conflicts_detected=[],
                    conflicts_resolved=0,
                    data_quality_score=0.0,
                    sources_processed=0,
                    processing_time=0.0,
                    errors=["No data sources available for aggregation"]
                )
            
            # Normalize data formats
            normalized_sources = await self._normalize_data_formats(data_sources)
            
            # Detect conflicts
            conflicts = await self._detect_data_conflicts(normalized_sources, date_range)
            
            # Resolve conflicts intelligently
            resolved_conflicts = await self._resolve_conflicts_intelligently(conflicts)
            
            # Aggregate data
            aggregated_data = await self._aggregate_data_by_strategy(
                normalized_sources, resolved_conflicts, aggregation_level, metrics
            )
            
            # Calculate overall data quality score
            quality_score = self._calculate_overall_quality_score(normalized_sources)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                normalized_sources, conflicts, quality_score
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AggregationResult(
                success=True,
                aggregated_data=aggregated_data,
                conflicts_detected=conflicts,
                conflicts_resolved=len(resolved_conflicts),
                data_quality_score=quality_score,
                sources_processed=len(data_sources),
                processing_time=processing_time,
                recommendations=recommendations
            )
            
            # Save aggregation results
            await self._save_aggregation_results(result, date_range)
            
            logger.info(f"Data aggregation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AggregationResult(
                success=False,
                aggregated_data=[],
                conflicts_detected=[],
                conflicts_resolved=0,
                data_quality_score=0.0,
                sources_processed=0,
                processing_time=processing_time,
                errors=[f"Aggregation failed: {str(e)}"]
            )
    
    async def _collect_data_sources(
        self,
        date_range: Tuple[date, date],
        platforms: Optional[List[str]] = None
    ) -> List[DataSource]:
        """Collect data from all connected marketing platforms"""
        data_sources = []
        
        try:
            # Get active platform connections
            query = text("""
                SELECT 
                    pc.id as connection_id,
                    pc.platform_id,
                    pc.platform_name,
                    pc.is_active,
                    pc.last_sync,
                    pc.data_quality_score,
                    pc.attribution_model,
                    pc.time_zone,
                    pc.currency,
                    pc.connection_config
                FROM platform_connections pc
                WHERE pc.is_active = true
                AND (:platforms IS NULL OR pc.platform_id = ANY(:platforms))
                ORDER BY pc.data_quality_score DESC
            """)
            
            result = self.db_session.execute(query, {
                'platforms': platforms
            })
            
            # Process each platform connection
            tasks = []
            for row in result:
                task = self._fetch_platform_data(row, date_range)
                tasks.append(task)
            
            # Execute data fetching concurrently
            if tasks:
                platform_data_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(platform_data_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching data from platform: {str(result)}")
                    elif result:
                        data_sources.append(result)
            
            logger.info(f"Collected data from {len(data_sources)} sources")
            return data_sources
            
        except Exception as e:
            logger.error(f"Error collecting data sources: {str(e)}")
            return []
    
    async def _fetch_platform_data(
        self,
        connection_row: Any,
        date_range: Tuple[date, date]
    ) -> Optional[DataSource]:
        """Fetch data from a specific platform"""
        try:
            platform_id = connection_row.platform_id
            platform_name = connection_row.platform_name
            
            # Query platform-specific data
            data_query = text(f"""
                SELECT 
                    date,
                    campaign_id,
                    campaign_name,
                    impressions,
                    clicks,
                    conversions,
                    spend,
                    revenue,
                    ctr,
                    cpc,
                    cpm,
                    roas,
                    conversion_rate,
                    frequency,
                    reach,
                    attribution_model,
                    data_source_timestamp
                FROM {platform_id}_marketing_data
                WHERE date BETWEEN :start_date AND :end_date
                AND connection_id = :connection_id
                ORDER BY date, campaign_id
            """)
            
            result = self.db_session.execute(data_query, {
                'start_date': date_range[0],
                'end_date': date_range[1],
                'connection_id': connection_row.connection_id
            })
            
            data = []
            for row in result:
                record = {
                    'date': row.date,
                    'campaign_id': row.campaign_id,
                    'campaign_name': row.campaign_name,
                    'impressions': row.impressions,
                    'clicks': row.clicks,
                    'conversions': row.conversions,
                    'spend': row.spend,
                    'revenue': row.revenue,
                    'ctr': row.ctr,
                    'cpc': row.cpc,
                    'cpm': row.cpm,
                    'roas': row.roas,
                    'conversion_rate': row.conversion_rate,
                    'frequency': row.frequency,
                    'reach': row.reach,
                    'attribution_model': row.attribution_model,
                    'data_source_timestamp': row.data_source_timestamp,
                    'platform_id': platform_id
                }
                data.append(record)
            
            return DataSource(
                platform_id=platform_id,
                platform_name=platform_name,
                connection_id=connection_row.connection_id,
                data_quality_score=connection_row.data_quality_score,
                last_sync=connection_row.last_sync,
                attribution_model=connection_row.attribution_model,
                time_zone=connection_row.time_zone,
                currency=connection_row.currency,
                data=data,
                metadata={
                    'connection_config': connection_row.connection_config,
                    'record_count': len(data)
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching data from {connection_row.platform_name}: {str(e)}")
            return None
    
    async def _normalize_data_formats(self, data_sources: List[DataSource]) -> List[DataSource]:
        """Normalize data formats across different platforms"""
        normalized_sources = []
        
        for source in data_sources:
            try:
                normalized_data = []
                
                for record in source.data:
                    # Normalize currency
                    normalized_record = await self._normalize_currency(record, source.currency)
                    
                    # Normalize time zones
                    normalized_record = await self._normalize_timezone(normalized_record, source.time_zone)
                    
                    # Normalize attribution models
                    normalized_record = await self._normalize_attribution(
                        normalized_record, source.attribution_model
                    )
                    
                    # Standardize metric names
                    normalized_record = self._standardize_metric_names(normalized_record)
                    
                    normalized_data.append(normalized_record)
                
                # Create normalized source
                normalized_source = DataSource(
                    platform_id=source.platform_id,
                    platform_name=source.platform_name,
                    connection_id=source.connection_id,
                    data_quality_score=source.data_quality_score,
                    last_sync=source.last_sync,
                    attribution_model="normalized_last_click",  # Standardized
                    time_zone="UTC",  # Standardized
                    currency="USD",  # Standardized
                    data=normalized_data,
                    metadata=source.metadata
                )
                
                normalized_sources.append(normalized_source)
                
            except Exception as e:
                logger.error(f"Error normalizing data for {source.platform_name}: {str(e)}")
                # Include original source if normalization fails
                normalized_sources.append(source)
        
        return normalized_sources
    
    async def _normalize_currency(self, record: Dict[str, Any], source_currency: str) -> Dict[str, Any]:
        """Normalize currency to USD"""
        if source_currency == "USD":
            return record
        
        # Get exchange rate (simplified - in production, use real-time rates)
        exchange_rates = {
            'EUR': 1.1, 'GBP': 1.3, 'CAD': 0.8, 'AUD': 0.7,
            'JPY': 0.009, 'CHF': 1.1, 'CNY': 0.15
        }
        
        rate = exchange_rates.get(source_currency, 1.0)
        
        # Convert monetary fields
        monetary_fields = ['spend', 'revenue', 'cpc', 'cpm']
        for field in monetary_fields:
            if field in record and record[field] is not None:
                record[field] = record[field] * rate
        
        return record
    
    async def _normalize_timezone(self, record: Dict[str, Any], source_timezone: str) -> Dict[str, Any]:
        """Normalize timezone to UTC"""
        # Simplified timezone normalization
        # In production, use proper timezone conversion libraries
        if source_timezone == "UTC":
            return record
        
        # For now, just mark that normalization occurred
        record['timezone_normalized'] = True
        record['original_timezone'] = source_timezone
        
        return record
    
    async def _normalize_attribution(self, record: Dict[str, Any], attribution_model: str) -> Dict[str, Any]:
        """Normalize attribution models"""
        # Attribution model conversion factors (simplified)
        attribution_factors = {
            'first_click': {'conversions': 1.2, 'revenue': 1.2},
            'last_click': {'conversions': 1.0, 'revenue': 1.0},
            'linear': {'conversions': 0.9, 'revenue': 0.9},
            'time_decay': {'conversions': 0.95, 'revenue': 0.95},
            'position_based': {'conversions': 1.05, 'revenue': 1.05}
        }
        
        factors = attribution_factors.get(attribution_model.lower(), {})
        
        for metric, factor in factors.items():
            if metric in record and record[metric] is not None:
                record[metric] = record[metric] * factor
        
        record['attribution_normalized'] = True
        record['original_attribution'] = attribution_model
        
        return record
    
    def _standardize_metric_names(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize metric names across platforms"""
        # Metric name mappings
        name_mappings = {
            'cost': 'spend',
            'amount_spent': 'spend',
            'total_spend': 'spend',
            'click_through_rate': 'ctr',
            'clickthrough_rate': 'ctr',
            'cost_per_click': 'cpc',
            'cost_per_mille': 'cpm',
            'cost_per_thousand': 'cpm',
            'return_on_ad_spend': 'roas',
            'conv_rate': 'conversion_rate',
            'cvr': 'conversion_rate'
        }
        
        standardized_record = {}
        for key, value in record.items():
            standardized_key = name_mappings.get(key.lower(), key)
            standardized_record[standardized_key] = value
        
        return standardized_record
    
    async def _detect_data_conflicts(
        self,
        data_sources: List[DataSource],
        date_range: Tuple[date, date]
    ) -> List[DataConflict]:
        """Detect conflicts between data sources"""
        conflicts = []
        
        try:
            # Group data by date and campaign
            date_campaign_data = {}
            
            for source in data_sources:
                for record in source.data:
                    key = (record['date'], record.get('campaign_id', 'unknown'))
                    
                    if key not in date_campaign_data:
                        date_campaign_data[key] = {}
                    
                    date_campaign_data[key][source.platform_id] = {
                        'record': record,
                        'source': source,
                        'quality_score': source.data_quality_score
                    }
            
            # Check for conflicts in each date/campaign combination
            for (date_val, campaign_id), platform_data in date_campaign_data.items():
                if len(platform_data) > 1:  # Multiple sources for same date/campaign
                    conflicts.extend(
                        await self._check_metric_conflicts(date_val, campaign_id, platform_data)
                    )
            
            logger.info(f"Detected {len(conflicts)} data conflicts")
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {str(e)}")
            return []
    
    async def _check_metric_conflicts(
        self,
        date_val: date,
        campaign_id: str,
        platform_data: Dict[str, Dict[str, Any]]
    ) -> List[DataConflict]:
        """Check for conflicts in specific metrics"""
        conflicts = []
        
        # Metrics to check for conflicts
        metrics_to_check = ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        
        for metric in metrics_to_check:
            values = {}
            confidence_scores = {}
            
            for platform_id, data in platform_data.items():
                record = data['record']
                if metric in record and record[metric] is not None:
                    values[platform_id] = record[metric]
                    confidence_scores[platform_id] = data['quality_score']
            
            if len(values) > 1:
                # Check for significant differences
                value_list = list(values.values())
                max_val = max(value_list)
                min_val = min(value_list)
                
                # Consider it a conflict if difference > 10% of max value
                if max_val > 0 and (max_val - min_val) / max_val > 0.1:
                    conflict = DataConflict(
                        conflict_type=ConflictType.VALUE_MISMATCH,
                        metric_name=metric,
                        date=date_val,
                        sources_involved=list(values.keys()),
                        values=values,
                        confidence_scores=confidence_scores,
                        resolution_strategy="",
                        resolved_value=None,
                        manual_review_required=False
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _resolve_conflicts_intelligently(self, conflicts: List[DataConflict]) -> List[DataConflict]:
        """Resolve conflicts using intelligent strategies"""
        resolved_conflicts = []
        
        for conflict in conflicts:
            try:
                # Determine resolution strategy based on conflict type and metric
                strategy = self._determine_resolution_strategy(conflict)
                conflict.resolution_strategy = strategy
                
                # Apply resolution strategy
                if strategy == "highest_quality":
                    resolved_value = await self._resolve_by_highest_quality(conflict)
                elif strategy == "weighted_average":
                    resolved_value = await self._resolve_by_weighted_average(conflict)
                elif strategy == "intelligent_merge":
                    resolved_value = await self._resolve_by_intelligent_merge(conflict)
                elif strategy == "manual_review":
                    resolved_value = None
                    conflict.manual_review_required = True
                else:
                    resolved_value = await self._resolve_by_latest_data(conflict)
                
                conflict.resolved_value = resolved_value
                resolved_conflicts.append(conflict)
                
            except Exception as e:
                logger.error(f"Error resolving conflict for {conflict.metric_name}: {str(e)}")
                conflict.manual_review_required = True
                resolved_conflicts.append(conflict)
        
        return resolved_conflicts
    
    def _determine_resolution_strategy(self, conflict: DataConflict) -> str:
        """Determine the best resolution strategy for a conflict"""
        metric = conflict.metric_name
        
        # Strategy based on metric type
        if metric in ['impressions', 'clicks', 'conversions', 'spend', 'revenue']:
            # For volume metrics, use highest quality source
            return "highest_quality"
        elif metric in ['ctr', 'cpc', 'cpm', 'roas', 'conversion_rate']:
            # For rate metrics, use weighted average
            return "weighted_average"
        elif metric == 'reach':
            # For reach, avoid double counting
            return "intelligent_merge"
        else:
            # Default to highest quality
            return "highest_quality"
    
    async def _resolve_by_highest_quality(self, conflict: DataConflict) -> Any:
        """Resolve conflict by selecting highest quality source"""
        best_source = max(conflict.confidence_scores.items(), key=lambda x: x[1])
        return conflict.values[best_source[0]]
    
    async def _resolve_by_weighted_average(self, conflict: DataConflict) -> float:
        """Resolve conflict by calculating weighted average"""
        total_weight = sum(conflict.confidence_scores.values())
        weighted_sum = sum(
            value * conflict.confidence_scores[source]
            for source, value in conflict.values.items()
        )
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    async def _resolve_by_intelligent_merge(self, conflict: DataConflict) -> Any:
        """Resolve conflict using AI-powered intelligent merging"""
        try:
            # Use KSE for intelligent conflict resolution
            context = {
                'metric': conflict.metric_name,
                'date': conflict.date.isoformat(),
                'sources': conflict.sources_involved,
                'values': conflict.values,
                'confidence_scores': conflict.confidence_scores
            }
            
            # Query KSE for resolution recommendation
            query = f"How should I resolve a {conflict.metric_name} conflict with values {conflict.values} from sources {conflict.sources_involved}?"
            
            kse_response = await self.kse_client.query(
                query=query,
                context=context,
                domain="marketing_data_aggregation"
            )
            
            if kse_response and 'resolved_value' in kse_response:
                return kse_response['resolved_value']
            else:
                # Fallback to weighted average
                return await self._resolve_by_weighted_average(conflict)
                
        except Exception as e:
            logger.error(f"Error in intelligent merge: {str(e)}")
            # Fallback to weighted average
            return await self._resolve_by_weighted_average(conflict)
    
    async def _resolve_by_latest_data(self, conflict: DataConflict) -> Any:
        """Resolve conflict by using latest data"""
        # Find source with most recent data
        latest_source = None
        latest_time = None
        
        for source in conflict.sources_involved:
            # This would need to be implemented based on actual data timestamps
            # For now, use first source
            if latest_source is None:
                latest_source = source
        
        return conflict.values.get(latest_source)
    
    async def _aggregate_data_by_strategy(
        self,
        data_sources: List[DataSource],
        resolved_conflicts: List[DataConflict],
        aggregation_level: str,
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate data using specified strategies"""
        aggregated_data = []
        
        try:
            # Create conflict resolution lookup
            conflict_resolutions = {}
            for conflict in resolved_conflicts:
                key = (conflict.date, conflict.metric_name)
                conflict_resolutions[key] = conflict.resolved_value
            
            # Group data by aggregation level
            if aggregation_level == 'daily':
                grouped_data = self._group_by_daily(data_sources)
            elif aggregation_level == 'weekly':
                grouped_data = self._group_by_weekly(data_sources)
            elif aggregation_level == 'monthly':
                grouped_data = self._group_by_monthly(data_sources)
            else:
                grouped_data = self._group_by_daily(data_sources)
            
            # Aggregate each group
            for date_key, records in grouped_data.items():
                aggregated_record = await self._aggregate_record_group(
                    date_key, records, conflict_resolutions, metrics
                )
                aggregated_data.append(aggregated_record)
            
            # Sort by date
            aggregated_data.sort(key=lambda x: x['date'])
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return []
    
    def _group_by_daily(self, data_sources: List[DataSource]) -> Dict[date, List[Dict[str, Any]]]:
        """Group data by daily periods"""
        grouped = {}
        
        for source in data_sources:
            for record in source.data:
                date_key = record['date']
                if date_key not in grouped:
                    grouped[date_key] = []
                
                # Add source information to record
                record_with_source = record.copy()
                record_with_source['source_platform'] = source.platform_id
                record_with_source['source_quality'] = source.data_quality_score
                
                grouped[date_key].append(record_with_source)
        
        return grouped
    
    def _group_by_weekly(self, data_sources: List[DataSource]) -> Dict[date, List[Dict[str, Any]]]:
        """Group data by weekly periods"""
        grouped = {}
        
        for source in data_sources:
            for record in source.data:
                # Get Monday of the week
                record_date = record['date']
                week_start = record_date - timedelta(days=record_date.weekday())
                
                if week_start not in grouped:
                    grouped[week_start] = []
                
                record_with_source = record.copy()
                record_with_source['source_platform'] = source.platform_id
                record_with_source['source_quality'] = source.data_quality_score
                
                grouped[week_start].append(record_with_source)
        
        return grouped
    
    def _group_by_monthly(self, data_sources: List[DataSource]) -> Dict[date, List[Dict[str, Any]]]:
        """Group data by monthly periods"""
        grouped = {}
        
        for source in data_sources:
            for record in source.data:
                # Get first day of the month
                record_date = record['date']
                month_start = date(record_date.year, record_date.month, 1)
                
                if month_start not in grouped:
                    grouped[month_start] = []
                
                record_with_source = record.copy()
                record_with_source['source_platform'] = source.platform_id
                record_with_source['source_quality'] = source.data_quality_score
                
                grouped[month_start].append(record_with_source)
        
        return grouped
    
    async def _aggregate_record_group(
        self,
        date_key: date,
        records: List[Dict[str, Any]],
        conflict_resolutions: Dict[Tuple[date, str], Any],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Aggregate a group of records for a specific date"""
        aggregated = {
            'date': date_key,
            'sources_count': len(set(r['source_platform'] for r in records)),
            'records_count': len(records),
            'data_quality_score': sum(r['source_quality'] for r in records) / len(records)
        }
        
        # Determine which metrics to aggregate
        all_metrics = set()
        for record in records:
            all_metrics.update(record.keys())
        
        metrics_to_aggregate = metrics or [
            m for m in all_metrics 
            if m in self.metric_strategies and m not in ['date', 'source_platform', 'source_quality']
        ]
        
        # Aggregate each metric
        for metric in metrics_to_aggregate:
            # Check for conflict resolution first
            conflict_key = (date_key, metric)
            if conflict_key in conflict_resolutions:
                aggregated[metric] = conflict_resolutions[conflict_key]
                continue
            
            # Get values for this metric
            values = []
            weights = []
            
            for record in records:
                if metric in record and record[metric] is not None:
                    values.append(record[metric])
                    weights.append(record['source_quality'])
            
            if not values:
                aggregated[metric] = None
                continue
            
            # Apply aggregation strategy
            strategy = self.metric_strategies.get(metric, AggregationStrategy.SUM)
            
            if strategy == AggregationStrategy.SUM:
                aggregated[metric] = sum(values)
            elif strategy == AggregationStrategy.AVERAGE:
                aggregated[metric] = sum(values) / len(values)
            elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                total_weight = sum(weights)
                if total_weight > 0:
                    aggregated[metric] = sum(v * w for v, w in zip(values, weights)) / total_weight
                else:
                    aggregated[metric] = sum(values) / len(values)
            elif strategy == AggregationStrategy.LATEST_WINS:
                aggregated[metric] = values[-1]  # Assuming records are sorted by time
            elif strategy == AggregationStrategy.HIGHEST_QUALITY:
                best_idx = weights.index(max(weights))
                aggregated[metric] = values[best_idx]
            elif strategy == AggregationStrategy.INTELLIGENT_MERGE:
                # For reach and similar metrics, avoid double counting
                if metric == 'reach':
                    aggregated[metric] = max(values)  # Take maximum reach
                else:
                    aggregated[metric] = sum(values)
            else:
                aggregated[metric] = sum(values)
        
        return aggregated
    
    def _calculate_overall_quality_score(self, data_sources: List[DataSource]) -> float:
        """Calculate overall data quality score"""
        if not data_sources:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for source in data_sources:
            weight = len(source.data)  # Weight by number of records
            total_score += source.data_quality_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def _generate_recommendations(
        self,
        data_sources: List[DataSource],
        conflicts: List[DataConflict],
        quality_score: float
    ) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.7:
            recommendations.append("Consider improving data quality by reviewing platform connections and data validation rules")
        
        # Conflict-based recommendations
        high_conflict_metrics = {}
        for conflict in conflicts:
            metric = conflict.metric_name
            high_conflict_metrics[metric] = high_conflict_metrics.get(metric, 0) + 1
        
        for metric, count in high_conflict_metrics.items():
            if count > 5:  # More than 5 conflicts for this metric
                recommendations.append(f"High number of conflicts detected for {metric}. Consider reviewing attribution models and data collection methods")
        
        # Source-based recommendations
        low_quality_sources = [s for s in data_sources if s.data_quality_score < 0.6]
        if low_quality_sources:
            source_names = [s.platform_name for s in low_quality_sources]
            recommendations.append(f"Low quality data detected from: {', '.join(source_names)}. Consider reconnecting or updating these platforms")
        
        # Missing data recommendations
        date_coverage = {}
        for source in data_sources:
            for record in source.data:
                date_key = record['date']
                if date_key not in date_coverage:
                    date_coverage[date_key] = set()
                date_coverage[date_key].add(source.platform_id)
        
        incomplete_dates = [date_key for date_key, sources in date_coverage.items() if len(sources) < len(data_sources)]
        if len(incomplete_dates) > len(date_coverage) * 0.1:  # More than 10% incomplete
            recommendations.append("Significant data gaps detected. Consider reviewing data sync schedules and platform connectivity")
        
        return recommendations
    
    async def _save_aggregation_results(self, result: AggregationResult, date_range: Tuple[date, date]) -> bool:
        """Save aggregation results to database"""
        try:
            # Save aggregated data
            for record in result.aggregated_data:
                insert_query = text("""
                    INSERT INTO aggregated_marketing_data 
                    (date, impressions, clicks, conversions, spend, revenue, ctr, cpc, cpm, roas, 
                     conversion_rate, frequency, reach, sources_count, records_count, data_quality_score,
                     aggregation_timestamp, aggregation_level)
                    VALUES 
                    (:date, :impressions, :clicks, :conversions, :spend, :revenue, :ctr, :cpc, :cpm, :roas,
                     :conversion_rate, :frequency, :reach, :sources_count, :records_count, :data_quality_score,
                     :aggregation_timestamp, :aggregation_level)
                    ON CONFLICT (date, aggregation_level) DO UPDATE SET
                        impressions = EXCLUDED.impressions,
                        clicks = EXCLUDED.clicks,
                        conversions = EXCLUDED.conversions,
                        spend = EXCLUDED.spend,
                        revenue = EXCLUDED.revenue,
                        ctr = EXCLUDED.ctr,
                        cpc = EXCLUDED.cpc,
                        cpm = EXCLUDED.cpm,
                        roas = EXCLUDED.roas,
                        conversion_rate = EXCLUDED.conversion_rate,
                        frequency = EXCLUDED.frequency,
                        reach = EXCLUDED.reach,
                        sources_count = EXCLUDED.sources_count,
                        records_count = EXCLUDED.records_count,
                        data_quality_score = EXCLUDED.data_quality_score,
                        aggregation_timestamp = EXCLUDED.aggregation_timestamp
                """)
                
                self.db_session.execute(insert_query, {
                    'date': record['date'],
                    'impressions': record.get('impressions'),
                    'clicks': record.get('clicks'),
                    'conversions': record.get('conversions'),
                    'spend': record.get('spend'),
                    'revenue': record.get('revenue'),
                    'ctr': record.get('ctr'),
                    'cpc': record.get('cpc'),
                    'cpm': record.get('cpm'),
                    'roas': record.get('roas'),
                    'conversion_rate': record.get('conversion_rate'),
                    'frequency': record.get('frequency'),
                    'reach': record.get('reach'),
                    'sources_count': record.get('sources_count'),
                    'records_count': record.get('records_count'),
                    'data_quality_score': record.get('data_quality_score'),
                    'aggregation_timestamp': datetime.now(),
                    'aggregation_level': 'daily'  # Default for now
                })
            
            # Save conflicts
            for conflict in result.conflicts_detected:
                conflict_query = text("""
                    INSERT INTO data_conflicts 
                    (date, metric_name, conflict_type, sources_involved, values, 
                     confidence_scores, resolution_strategy, resolved_value, 
                     manual_review_required, detection_timestamp)
                    VALUES 
                    (:date, :metric_name, :conflict_type, :sources_involved, :values,
                     :confidence_scores, :resolution_strategy, :resolved_value,
                     :manual_review_required, :detection_timestamp)
                """)
                
                self.db_session.execute(conflict_query, {
                    'date': conflict.date,
                    'metric_name': conflict.metric_name,
                    'conflict_type': conflict.conflict_type.value,
                    'sources_involved': json.dumps(conflict.sources_involved),
                    'values': json.dumps(conflict.values),
                    'confidence_scores': json.dumps(conflict.confidence_scores),
                    'resolution_strategy': conflict.resolution_strategy,
                    'resolved_value': conflict.resolved_value,
                    'manual_review_required': conflict.manual_review_required,
                    'detection_timestamp': datetime.now()
                })
            
            # Save aggregation metadata
            metadata_query = text("""
                INSERT INTO aggregation_log 
                (start_date, end_date, sources_processed, conflicts_resolved, 
                 data_quality_score, processing_time, recommendations, 
                 aggregation_timestamp)
                VALUES 
                (:start_date, :end_date, :sources_processed, :conflicts_resolved,
                 :data_quality_score, :processing_time, :recommendations,
                 :aggregation_timestamp)
            """)
            
            self.db_session.execute(metadata_query, {
                'start_date': date_range[0],
                'end_date': date_range[1],
                'sources_processed': result.sources_processed,
                'conflicts_resolved': result.conflicts_resolved,
                'data_quality_score': result.data_quality_score,
                'processing_time': result.processing_time,
                'recommendations': json.dumps(result.recommendations),
                'aggregation_timestamp': datetime.now()
            })
            
            self.db_session.commit()
            logger.info("Aggregation results saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving aggregation results: {str(e)}")
            self.db_session.rollback()
            return False
    
    def get_aggregation_status(self) -> Dict[str, Any]:
        """Get current aggregation status and metrics"""
        try:
            # Get latest aggregation info
            query = text("""
                SELECT 
                    aggregation_timestamp,
                    start_date,
                    end_date,
                    sources_processed,
                    conflicts_resolved,
                    data_quality_score,
                    processing_time
                FROM aggregation_log 
                ORDER BY aggregation_timestamp DESC 
                LIMIT 1
            """)
            
            result = self.db_session.execute(query)
            row = result.fetchone()
            
            if row:
                return {
                    'last_aggregation': row.aggregation_timestamp,
                    'date_range': f"{row.start_date} to {row.end_date}",
                    'sources_processed': row.sources_processed,
                    'conflicts_resolved': row.conflicts_resolved,
                    'data_quality_score': row.data_quality_score,
                    'processing_time': row.processing_time,
                    'status': 'active'
                }
            else:
                return {
                    'status': 'not_initialized',
                    'message': 'No aggregation has been performed yet'
                }
                
        except Exception as e:
            logger.error(f"Error getting aggregation status: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error retrieving status: {str(e)}"
            }
    
    async def get_conflict_summary(self, date_range: Optional[Tuple[date, date]] = None) -> Dict[str, Any]:
        """Get summary of data conflicts"""
        try:
            where_clause = ""
            params = {}
            
            if date_range:
                where_clause = "WHERE date BETWEEN :start_date AND :end_date"
                params = {'start_date': date_range[0], 'end_date': date_range[1]}
            
            query = text(f"""
                SELECT 
                    conflict_type,
                    metric_name,
                    COUNT(*) as conflict_count,
                    AVG(CASE WHEN manual_review_required THEN 1 ELSE 0 END) as manual_review_rate,
                    COUNT(DISTINCT date) as affected_dates
                FROM data_conflicts 
                {where_clause}
                GROUP BY conflict_type, metric_name
                ORDER BY conflict_count DESC
            """)
            
            result = self.db_session.execute(query, params)
            
            conflicts_summary = []
            for row in result:
                conflicts_summary.append({
                    'conflict_type': row.conflict_type,
                    'metric_name': row.metric_name,
                    'conflict_count': row.conflict_count,
                    'manual_review_rate': row.manual_review_rate,
                    'affected_dates': row.affected_dates
                })
            
            return {
                'conflicts_by_type': conflicts_summary,
                'total_conflicts': sum(c['conflict_count'] for c in conflicts_summary),
                'metrics_affected': len(set(c['metric_name'] for c in conflicts_summary))
            }
            
        except Exception as e:
            logger.error(f"Error getting conflict summary: {str(e)}")
            return {'error': str(e)}
    
    async def schedule_automatic_aggregation(
        self,
        schedule_type: str = 'daily',
        time_of_day: str = '02:00'
    ) -> bool:
        """Schedule automatic data aggregation"""
        try:
            # This would integrate with a job scheduler like Celery or APScheduler
            # For now, just log the scheduling request
            logger.info(f"Scheduling automatic aggregation: {schedule_type} at {time_of_day}")
            
            # Save schedule configuration
            schedule_query = text("""
                INSERT INTO aggregation_schedule 
                (schedule_type, time_of_day, is_active, created_at)
                VALUES 
                (:schedule_type, :time_of_day, :is_active, :created_at)
                ON CONFLICT (schedule_type) DO UPDATE SET
                    time_of_day = EXCLUDED.time_of_day,
                    is_active = EXCLUDED.is_active,
                    updated_at = EXCLUDED.created_at
            """)
            
            self.db_session.execute(schedule_query, {
                'schedule_type': schedule_type,
                'time_of_day': time_of_day,
                'is_active': True,
                'created_at': datetime.now()
            })
            
            self.db_session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling aggregation: {str(e)}")
            return False
        for record in records:
            all_metrics.update(record.keys())
        
        metrics_to_aggregate = metrics or [
            m for m in all_metrics 
            if m in self.metric_strategies and m not in ['date', 'source_platform', 'source_quality']
        ]
        
        # Aggregate each metric
        for metric in metrics_to_aggregate:
            # Check for conflict resolution first
            conflict_key = (date_key, metric)
            if conflict_key in conflict_resolutions:
                aggregated[metric] = conflict_resolutions[conflict_key]
                continue
            
            # Get values for this metric
            values = []
            weights = []
            
            for record in records:
                if metric in record and record[metric] is not None:
                    values.append(record[metric])
                    weights.append(record['source_quality'])
            
            if not values:
                aggregated[metric] = None
                continue
            