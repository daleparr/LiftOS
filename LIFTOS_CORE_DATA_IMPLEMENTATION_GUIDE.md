# LiftOS Core Data Implementation Guide
*Practical Implementation Steps for Centralized Data Architecture*

## Overview

This guide provides detailed implementation steps for moving the data architecture from the causal microservice to LiftOS Core, leveraging existing KSE memory infrastructure for maximum efficiency and minimal risk.

## Implementation Components

### 1. Enhanced Calendar Dimension Engine

```python
# File: shared/utils/calendar_engine.py
"""
Enhanced Calendar Dimension Engine for LiftOS Core
Provides comprehensive calendar features for causal modeling
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from enum import Enum
import holidays

class Industry(str, Enum):
    """Industry types for seasonality patterns"""
    RETAIL = "retail"
    B2B_SAAS = "b2b_saas"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    TRAVEL = "travel"
    GENERAL = "general"

class Region(str, Enum):
    """Regional markets for calendar adjustments"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"

class CalendarDimensionEngine:
    """Advanced calendar dimension engine with industry-specific patterns"""
    
    def __init__(self, tenant_config: Dict[str, Any]):
        self.tenant_id = tenant_config['tenant_id']
        self.industry = Industry(tenant_config.get('industry', 'general'))
        self.region = Region(tenant_config.get('region', 'north_america'))
        self.fiscal_year_start = tenant_config.get('fiscal_year_start', 1)  # January
        self.custom_events = tenant_config.get('custom_events', {})
        
        # Initialize holiday calendars
        self.holiday_calendar = self._initialize_holiday_calendar()
        
        # Load seasonality patterns
        self.seasonality_patterns = self._load_seasonality_patterns()
    
    async def enrich_with_calendar(self, target_date: date) -> Dict[str, Any]:
        """Enrich data with comprehensive calendar dimensions"""
        
        calendar_features = {
            # Basic temporal features
            'date_key': int(target_date.strftime('%Y%m%d')),
            'full_date': target_date.isoformat(),
            'day_of_week': target_date.weekday(),
            'day_name': target_date.strftime('%A'),
            'day_of_month': target_date.day,
            'day_of_year': target_date.timetuple().tm_yday,
            'week_of_year': target_date.isocalendar()[1],
            'month_number': target_date.month,
            'month_name': target_date.strftime('%B'),
            'quarter_number': (target_date.month - 1) // 3 + 1,
            'year_number': target_date.year,
            
            # Business calendar
            'is_weekday': target_date.weekday() < 5,
            'is_weekend': target_date.weekday() >= 5,
            'is_holiday': await self.is_holiday(target_date),
            'holiday_name': await self.get_holiday_name(target_date),
            
            # Fiscal calendar
            'fiscal_year': await self.get_fiscal_year(target_date),
            'fiscal_quarter': await self.get_fiscal_quarter(target_date),
            'fiscal_month': await self.get_fiscal_month(target_date),
            
            # Seasonality factors
            'week_seasonality_factor': await self.get_week_seasonality(target_date),
            'month_seasonality_factor': await self.get_month_seasonality(target_date),
            'quarter_seasonality_factor': await self.get_quarter_seasonality(target_date),
            
            # Marketing calendar
            'is_black_friday': await self.is_black_friday(target_date),
            'is_cyber_monday': await self.is_cyber_monday(target_date),
            'is_prime_day': await self.is_prime_day(target_date),
            'marketing_season': await self.get_marketing_season(target_date),
            
            # Industry-specific features
            'industry_seasonality': await self.get_industry_seasonality(target_date),
            'regional_adjustments': await self.get_regional_adjustments(target_date),
            
            # Custom events
            'custom_events': await self.get_custom_events(target_date)
        }
        
        return calendar_features
    
    def _initialize_holiday_calendar(self):
        """Initialize holiday calendar based on region"""
        
        holiday_calendars = {
            Region.NORTH_AMERICA: holidays.UnitedStates(),
            Region.EUROPE: holidays.Germany(),  # Can be customized per country
            Region.ASIA_PACIFIC: holidays.Japan(),  # Can be customized per country
            Region.LATIN_AMERICA: holidays.Mexico(),
            Region.MIDDLE_EAST: holidays.SaudiArabia()
        }
        
        return holiday_calendars.get(self.region, holidays.UnitedStates())
    
    def _load_seasonality_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load industry-specific seasonality patterns"""
        
        patterns = {
            Industry.RETAIL: {
                'q1_factor': 0.7,   # Post-holiday lull
                'q2_factor': 0.9,   # Spring season
                'q3_factor': 1.1,   # Back-to-school
                'q4_factor': 1.4,   # Holiday shopping
                'black_friday_boost': 2.5,
                'cyber_monday_boost': 2.1,
                'summer_dip': 0.8
            },
            Industry.B2B_SAAS: {
                'q1_factor': 1.3,   # New year planning
                'q2_factor': 1.1,   # Mid-year push
                'q3_factor': 0.9,   # Summer slowdown
                'q4_factor': 1.1,   # Budget flush
                'month_end_boost': 1.15,
                'summer_dip': 0.9
            },
            Industry.EDUCATION: {
                'q1_factor': 1.2,   # Spring semester
                'q2_factor': 0.8,   # End of school year
                'q3_factor': 1.5,   # Fall semester prep
                'q4_factor': 0.6,   # Winter break
                'back_to_school_boost': 1.8
            },
            Industry.HEALTHCARE: {
                'q1_factor': 1.2,   # New year health resolutions
                'q2_factor': 1.0,   # Steady
                'q3_factor': 0.9,   # Summer vacation impact
                'q4_factor': 0.8,   # Holiday season slowdown
                'flu_season_boost': 1.3
            },
            Industry.FINANCE: {
                'q1_factor': 1.4,   # Tax season
                'q2_factor': 1.0,   # Steady
                'q3_factor': 0.9,   # Summer slowdown
                'q4_factor': 1.2,   # Year-end planning
                'tax_season_boost': 1.6
            },
            Industry.TRAVEL: {
                'q1_factor': 0.8,   # Post-holiday lull
                'q2_factor': 1.3,   # Spring travel
                'q3_factor': 1.5,   # Summer peak
                'q4_factor': 1.1,   # Holiday travel
                'summer_peak': 1.8,
                'holiday_boost': 1.4
            }
        }
        
        return patterns.get(self.industry, patterns[Industry.GENERAL])
    
    async def is_holiday(self, target_date: date) -> bool:
        """Check if date is a holiday"""
        return target_date in self.holiday_calendar
    
    async def get_holiday_name(self, target_date: date) -> Optional[str]:
        """Get holiday name if date is a holiday"""
        return self.holiday_calendar.get(target_date)
    
    async def get_fiscal_year(self, target_date: date) -> int:
        """Calculate fiscal year based on fiscal year start month"""
        if target_date.month >= self.fiscal_year_start:
            return target_date.year
        else:
            return target_date.year - 1
    
    async def get_fiscal_quarter(self, target_date: date) -> int:
        """Calculate fiscal quarter"""
        fiscal_month = await self.get_fiscal_month(target_date)
        return (fiscal_month - 1) // 3 + 1
    
    async def get_fiscal_month(self, target_date: date) -> int:
        """Calculate fiscal month"""
        calendar_month = target_date.month
        fiscal_month = calendar_month - self.fiscal_year_start + 1
        if fiscal_month <= 0:
            fiscal_month += 12
        return fiscal_month
    
    async def get_week_seasonality(self, target_date: date) -> float:
        """Calculate week-based seasonality factor"""
        day_of_week = target_date.weekday()
        
        # Base weekly pattern (Monday=0, Sunday=6)
        weekly_factors = {
            0: 1.1,  # Monday - strong start
            1: 1.2,  # Tuesday - peak
            2: 1.15, # Wednesday - strong
            3: 1.1,  # Thursday - good
            4: 0.9,  # Friday - weekend prep
            5: 0.7,  # Saturday - weekend
            6: 0.6   # Sunday - weekend
        }
        
        base_factor = weekly_factors.get(day_of_week, 1.0)
        
        # Apply industry adjustments
        if self.industry == Industry.B2B_SAAS and day_of_week >= 5:
            base_factor *= 0.5  # B2B drops significantly on weekends
        elif self.industry == Industry.RETAIL and day_of_week >= 5:
            base_factor *= 1.3  # Retail peaks on weekends
        
        return base_factor
    
    async def get_month_seasonality(self, target_date: date) -> float:
        """Calculate month-based seasonality factor"""
        month = target_date.month
        quarter = (month - 1) // 3 + 1
        
        # Get base quarterly factor
        quarter_key = f'q{quarter}_factor'
        base_factor = self.seasonality_patterns.get(quarter_key, 1.0)
        
        # Apply month-specific adjustments
        month_adjustments = {
            1: 1.0,   # January baseline
            2: 0.95,  # February dip
            3: 1.05,  # March recovery
            4: 1.0,   # April baseline
            5: 1.05,  # May boost
            6: 0.95,  # June dip
            7: 0.9,   # July summer
            8: 1.1,   # August back-to-school
            9: 1.05,  # September strong
            10: 1.0,  # October baseline
            11: 1.2,  # November pre-holiday
            12: 1.3   # December holiday peak
        }
        
        month_factor = month_adjustments.get(month, 1.0)
        
        return base_factor * month_factor
    
    async def get_quarter_seasonality(self, target_date: date) -> float:
        """Calculate quarter-based seasonality factor"""
        quarter = (target_date.month - 1) // 3 + 1
        quarter_key = f'q{quarter}_factor'
        return self.seasonality_patterns.get(quarter_key, 1.0)
    
    async def is_black_friday(self, target_date: date) -> bool:
        """Check if date is Black Friday (4th Thursday of November + 1 day)"""
        if target_date.month != 11:
            return False
        
        # Find 4th Thursday of November
        first_day = date(target_date.year, 11, 1)
        first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
        fourth_thursday = first_thursday + timedelta(days=21)
        black_friday = fourth_thursday + timedelta(days=1)
        
        return target_date == black_friday
    
    async def is_cyber_monday(self, target_date: date) -> bool:
        """Check if date is Cyber Monday (Monday after Black Friday)"""
        if target_date.month != 11:
            return False
        
        black_friday_date = await self.get_black_friday_date(target_date.year)
        cyber_monday = black_friday_date + timedelta(days=3)
        
        return target_date == cyber_monday
    
    async def is_prime_day(self, target_date: date) -> bool:
        """Check if date is Amazon Prime Day (typically mid-July)"""
        # Prime Day dates vary by year, this is a simplified check
        return (target_date.month == 7 and 
                target_date.day >= 10 and 
                target_date.day <= 16)
    
    async def get_marketing_season(self, target_date: date) -> str:
        """Determine marketing season for the date"""
        month = target_date.month
        
        if month in [12, 1, 2]:
            return "holiday_winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall_holiday_prep"
        else:
            return "general"
    
    async def get_industry_seasonality(self, target_date: date) -> Dict[str, float]:
        """Get industry-specific seasonality adjustments"""
        month = target_date.month
        day_of_week = target_date.weekday()
        
        industry_factors = {}
        
        if self.industry == Industry.RETAIL:
            if await self.is_black_friday(target_date):
                industry_factors['black_friday_boost'] = 2.5
            elif await self.is_cyber_monday(target_date):
                industry_factors['cyber_monday_boost'] = 2.1
            elif month in [6, 7]:
                industry_factors['summer_dip'] = 0.8
        
        elif self.industry == Industry.B2B_SAAS:
            if target_date.day >= 25:  # Month end
                industry_factors['month_end_boost'] = 1.15
            if month in [6, 7, 8]:
                industry_factors['summer_dip'] = 0.9
        
        elif self.industry == Industry.EDUCATION:
            if month in [8, 9]:
                industry_factors['back_to_school_boost'] = 1.8
            elif month in [6, 7]:
                industry_factors['summer_break'] = 0.3
        
        return industry_factors
    
    async def get_regional_adjustments(self, target_date: date) -> Dict[str, float]:
        """Get region-specific calendar adjustments"""
        month = target_date.month
        regional_factors = {}
        
        if self.region == Region.NORTH_AMERICA:
            if await self.is_black_friday(target_date):
                regional_factors['black_friday'] = 2.1
            elif await self.is_cyber_monday(target_date):
                regional_factors['cyber_monday'] = 1.8
            elif target_date.month == 5 and target_date.weekday() == 0:  # Memorial Day
                regional_factors['memorial_day'] = 0.8
        
        elif self.region == Region.EUROPE:
            if month == 8:
                regional_factors['august_vacation'] = 0.5
            elif month == 12 and target_date.day == 26:
                regional_factors['boxing_day'] = 1.3
        
        elif self.region == Region.ASIA_PACIFIC:
            # Chinese New Year (simplified - typically late January/early February)
            if month in [1, 2]:
                regional_factors['chinese_new_year'] = 0.3
            # Singles Day (November 11)
            elif month == 11 and target_date.day == 11:
                regional_factors['singles_day'] = 2.5
        
        return regional_factors
    
    async def get_custom_events(self, target_date: date) -> List[str]:
        """Get custom events for the tenant on this date"""
        date_str = target_date.isoformat()
        return self.custom_events.get(date_str, [])
    
    async def get_black_friday_date(self, year: int) -> date:
        """Calculate Black Friday date for given year"""
        first_day = date(year, 11, 1)
        first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
        fourth_thursday = first_thursday + timedelta(days=21)
        return fourth_thursday + timedelta(days=1)
    
    async def generate_calendar_features_batch(
        self, 
        date_range: List[date]
    ) -> List[Dict[str, Any]]:
        """Generate calendar features for a batch of dates efficiently"""
        
        calendar_features_batch = []
        
        for target_date in date_range:
            features = await self.enrich_with_calendar(target_date)
            calendar_features_batch.append(features)
        
        return calendar_features_batch
```

### 2. Unified Data API Layer

```python
# File: shared/api/unified_data_api.py
"""
Unified Data API for Cross-Module Access
Provides standardized data access patterns for all LiftOS modules
"""
import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from enum import Enum

from shared.kse_sdk.client import kse_client
from shared.models.marketing import DataSource, MarketingDataEntry
from shared.utils.calendar_engine import CalendarDimensionEngine

class DataAccessPattern(str, Enum):
    """Data access patterns for different use cases"""
    ATTRIBUTION_MODELING = "attribution_modeling"
    INSIGHT_DISCOVERY = "insight_discovery"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CONTEXT_ENRICHMENT = "context_enrichment"
    PERFORMANCE_BENCHMARKING = "performance_benchmarking"

class UnifiedDataAPI:
    """Standardized data access layer for all LiftOS modules"""
    
    def __init__(self, module_name: str, tenant_config: Dict[str, Any]):
        self.module_name = module_name
        self.tenant_id = tenant_config['tenant_id']
        self.kse_client = kse_client
        self.calendar_engine = CalendarDimensionEngine(tenant_config)
        
    async def get_attribution_ready_data(
        self,
        date_range: Tuple[date, date],
        channels: Optional[List[str]] = None,
        include_calendar: bool = True,
        include_quality_filter: bool = True,
        min_quality_score: float = 0.8
    ) -> pd.DataFrame:
        """Get data formatted for causal attribution modeling"""
        
        # Build search query
        query_parts = [
            "attribution ready marketing data",
            f"from {date_range[0]} to {date_range[1]}"
        ]
        
        if channels:
            query_parts.append(f"channels: {', '.join(channels)}")
        
        query = " ".join(query_parts)
        
        # Search KSE memory
        search_results = await self.kse_client.search_memory(
            query=query,
            context=f"marketing_data_{self.tenant_id}",
            search_type="hybrid",
            filters={
                "tenant_id": self.tenant_id,
                "tags": ["marketing", "causal_ready"],
                "date_range": {
                    "start": date_range[0].isoformat(),
                    "end": date_range[1].isoformat()
                }
            },
            limit=10000
        )
        
        # Transform to attribution format
        attribution_records = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            
            # Apply quality filter
            if include_quality_filter:
                quality_score = metadata.get('quality_score', 0)
                if quality_score < min_quality_score:
                    continue
            
            # Extract core attribution features
            record = {
                'date': metadata.get('date'),
                'channel': metadata.get('channel'),
                'campaign_id': metadata.get('campaign_id'),
                'spend': metadata.get('spend', 0),
                'impressions': metadata.get('impressions', 0),
                'clicks': metadata.get('clicks', 0),
                'conversions': metadata.get('conversions', 0),
                'revenue': metadata.get('revenue', 0),
                
                # Derived metrics
                'cpm': metadata.get('cpm', 0),
                'cpc': metadata.get('cpc', 0),
                'ctr': metadata.get('ctr', 0),
                'cvr': metadata.get('cvr', 0),
                'roas': metadata.get('roas', 0),
                
                # Data quality
                'quality_score': metadata.get('quality_score', 0),
                'data_source': metadata.get('data_source')
            }
            
            # Add calendar features if requested
            if include_calendar:
                calendar_dims = metadata.get('calendar_dimensions', {})
                record.update({
                    'day_of_week': calendar_dims.get('day_of_week'),
                    'is_weekend': calendar_dims.get('is_weekend'),
                    'is_holiday': calendar_dims.get('is_holiday'),
                    'month_seasonality_factor': calendar_dims.get('month_seasonality_factor'),
                    'quarter_seasonality_factor': calendar_dims.get('quarter_seasonality_factor'),
                    'marketing_season': calendar_dims.get('marketing_season'),
                    'industry_seasonality': calendar_dims.get('industry_seasonality', {})
                })
            
            attribution_records.append(record)
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(attribution_records)
        
        if not df.empty:
            # Ensure proper data types
            df['date'] = pd.to_datetime(df['date'])
            numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue',
                             'cpm', 'cpc', 'ctr', 'cvr', 'roas', 'quality_score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    async def get_surfacing_insights(
        self,
        insight_type: str,
        context_window_days: int = 30,
        similarity_threshold: float = 0.7,
        max_insights: int = 50
    ) -> List[Dict[str, Any]]:
        """Get data optimized for surfacing insights"""
        
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=context_window_days)
        
        query = f"surfacing insights {insight_type} marketing performance trends"
        
        search_results = await self.kse_client.search_memory(
            query=query,
            context=f"marketing_data_{self.tenant_id}",
            search_type="conceptual",
            filters={
                "tenant_id": self.tenant_id,
                "tags": ["marketing", "insights"],
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            limit=max_insights * 2  # Get more to filter by similarity
        )
        
        # Transform for surfacing analysis
        surfacing_insights = []
        
        for result in search_results:
            similarity_score = result.get('similarity_score', 0)
            
            # Apply similarity threshold
            if similarity_score < similarity_threshold:
                continue
            
            metadata = result.get('metadata', {})
            
            insight = {
                'insight_category': insight_type,
                'insight_text': result.get('content', ''),
                'confidence_score': similarity_score,
                'data_source': metadata.get('data_source'),
                'channel': metadata.get('channel'),
                'date': metadata.get('date'),
                'key_metrics': {
                    'spend': metadata.get('spend', 0),
                    'impressions': metadata.get('impressions', 0),
                    'clicks': metadata.get('clicks', 0),
                    'conversions': metadata.get('conversions', 0),
                    'roas': metadata.get('roas', 0)
                },
                'temporal_context': {
                    'marketing_season': metadata.get('calendar_dimensions', {}).get('marketing_season'),
                    'is_weekend': metadata.get('calendar_dimensions', {}).get('is_weekend'),
                    'is_holiday': metadata.get('calendar_dimensions', {}).get('is_holiday')
                },
                'cross_channel_context': await self.extract_cross_channel_context(metadata),
                'quality_indicators': {
                    'data_quality_score': metadata.get('quality_score', 0),
                    'data_freshness_hours': self.calculate_data_freshness(metadata.get('ingestion_timestamp'))
                }
            }
            
            surfacing_insights.append(insight)
        
        # Sort by confidence score and limit results
        surfacing_insights.sort(key=lambda x: x['confidence_score'], reverse=True)
        return surfacing_insights[:max_insights]
    
    async def get_llm_context_data(
        self,
        user_query: str,
        context_limit: int = 10,
        include_recent_trends: bool = True,
        max_context_age_days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get relevant data context for LLM responses"""
        
        # Use semantic search to find relevant marketing context
        context_results = await self.kse_client.search_memory(
            query=user_query,
            context=f"marketing_data_{self.tenant_id}",
            search_type="neural",
            filters={
                "tenant_id": self.tenant_id,
                "tags": ["marketing"]
            },
            limit=context_limit * 2  # Get more to filter and rank
        )
        
        # Add recent trends if requested
        if include_recent_trends:
            recent_trends = await self.get_recent_trends(max_context_age_days)
            context_results.extend(recent_trends)
        
        # Format for LLM consumption
        llm_context = []
        
        for result in context_results:
            metadata = result.get('metadata', {})
            
            context_entry = {
                'relevance_score': result.get('similarity_score', 0),
                'content_summary': self.summarize_for_llm(result),
                'key_metrics': {
                    'channel': metadata.get('channel'),
                    'spend': metadata.get('spend', 0),
                    'roas': metadata.get('roas', 0),
                    'conversions': metadata.get('conversions', 0)
                },
                'temporal_context': {
                    'date': metadata.get('date'),
                    'marketing_season': metadata.get('calendar_dimensions', {}).get('marketing_season'),
                    'relative_time': self.calculate_relative_time(metadata.get('date'))
                },
                'data_quality': {
                    'quality_score': metadata.get('quality_score', 0),
                    'data_source': metadata.get('data_source'),
                    'confidence_level': self.calculate_confidence_level(metadata)
                },
                'actionable_insights': await self.extract_actionable_insights(metadata)
            }
            
            llm_context.append(context_entry)
        
        # Sort by relevance and limit
        llm_context.sort(key=lambda x: x['relevance_score'], reverse=True)
        return llm_context[:context_limit]
    
    async def get_trend_analysis_data(
        self,
        metric: str,
        time_period_days: int = 30,
        comparison_period_days: int = 30,
        granularity: str = "daily"
    ) -> Dict[str, Any]:
        """Get data for trend analysis with period-over-period comparison"""
        
        end_date = datetime.utcnow().date()
        current_start = end_date - timedelta(days=time_period_days)
        comparison_start = current_start - timedelta(days=comparison_period_days)
        
        # Get current period data
        current_data = await self.get_attribution_ready_data(
            date_range=(current_start, end_date),
            include_calendar=True
        )
        
        # Get comparison period data
        comparison_data = await self.get_attribution_ready_data(
            date_range=(comparison_start, current_start),
            include_calendar=True
        )
        
        # Analyze trends
        trend_analysis = {
            'metric': metric,
            'current_period': {
                'start_date': current_start.isoformat(),
                'end_date': end_date.isoformat(),
                'data_points': len(current_data),
                'total_value': current_data[metric].sum() if metric in current_data.columns else 0,
                'average_value': current_data[metric].mean() if metric in current_data.columns else 0,
                'trend_direction': self.calculate_trend_direction(current_data, metric)
            },
            'comparison_period': {
                'start_date': comparison_start.isoformat(),
                'end_date': current_start.isoformat(),
                'data_points': len(comparison_data),
                'total_value': comparison_data[metric].sum() if metric in comparison_data.columns else 0,
                'average_value': comparison_data[metric].mean() if metric in comparison_data.columns else 0
            },
            'period_over_period': self.calculate_period_over_period_change(
                current_data, comparison_data, metric
            ),
            'seasonality_analysis': await self.analyze_seasonality_impact(
                current_data, metric
            ),
            'anomalies': await self.detect_trend_anomalies(current_data, metric)
        }
        
        return trend_analysis
    
    async def get_performance_benchmarks(
        self,
        benchmark_type: str = "industry",
        channels: Optional[List[str]] = None,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get performance benchmarking data"""
        
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=time_period_days)