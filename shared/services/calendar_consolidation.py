"""
Calendar Dimension Consolidation Service
Unifies multiple calendar models into a single master dimension
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from shared.models.calendar_dimension import CalendarDimension
from shared.models.causal_marketing import CausalMarketingCalendar
from shared.models.marketing import MarketingCalendar
from shared.models.executive_calendar import (
    ExecutiveCalendarConfig, BusinessEvent, EventType, EventImpact
)
from shared.utils.config import get_service_config
from shared.utils.database import get_database_session

logger = logging.getLogger(__name__)

class ConsolidationStrategy(Enum):
    """Calendar consolidation strategies"""
    MERGE_ALL = "merge_all"  # Merge all calendar models
    EXECUTIVE_PRIORITY = "executive_priority"  # Executive config takes priority
    CAUSAL_PRIORITY = "causal_priority"  # Causal marketing takes priority
    CUSTOM_HIERARCHY = "custom_hierarchy"  # Custom priority hierarchy

class ConflictResolution(Enum):
    """How to resolve conflicts between calendar models"""
    LATEST_WINS = "latest_wins"  # Most recent data wins
    HIGHEST_PRIORITY = "highest_priority"  # Highest priority source wins
    MERGE_ATTRIBUTES = "merge_attributes"  # Merge non-conflicting attributes
    MANUAL_REVIEW = "manual_review"  # Flag for manual review

@dataclass
class CalendarSource:
    """Represents a calendar data source"""
    name: str
    model_type: str
    priority: int
    last_updated: datetime
    record_count: int
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsolidationResult:
    """Result of calendar consolidation"""
    success: bool
    master_calendar: List[Dict[str, Any]]
    conflicts_resolved: int
    records_merged: int
    sources_processed: int
    consolidation_metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class CalendarConsolidationService:
    """Service for consolidating multiple calendar dimensions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_service_config()
        self.db_session = get_database_session()
        self.consolidation_strategy = ConsolidationStrategy.EXECUTIVE_PRIORITY
        self.conflict_resolution = ConflictResolution.MERGE_ATTRIBUTES
        
        # Source priority mapping (higher number = higher priority)
        self.source_priorities = {
            'executive_calendar': 100,
            'causal_marketing': 80,
            'calendar_dimension': 60,
            'marketing_calendar': 40,
            'external_sources': 20
        }
    
    def consolidate_calendars(
        self,
        executive_config: Optional[ExecutiveCalendarConfig] = None,
        date_range: Optional[Tuple[date, date]] = None,
        strategy: Optional[ConsolidationStrategy] = None
    ) -> ConsolidationResult:
        """
        Consolidate all calendar sources into master dimension
        
        Args:
            executive_config: Executive calendar configuration
            date_range: Date range to consolidate (start_date, end_date)
            strategy: Consolidation strategy to use
            
        Returns:
            ConsolidationResult with consolidated calendar data
        """
        try:
            logger.info("Starting calendar consolidation process")
            
            # Set strategy
            if strategy:
                self.consolidation_strategy = strategy
            
            # Set default date range if not provided
            if not date_range:
                current_year = datetime.now().year
                date_range = (date(current_year, 1, 1), date(current_year + 1, 12, 31))
            
            # Collect all calendar sources
            sources = self._collect_calendar_sources(executive_config, date_range)
            
            # Validate sources
            if not sources:
                return ConsolidationResult(
                    success=False,
                    master_calendar=[],
                    conflicts_resolved=0,
                    records_merged=0,
                    sources_processed=0,
                    consolidation_metadata={},
                    errors=["No calendar sources found to consolidate"]
                )
            
            # Generate master calendar structure
            master_calendar = self._generate_master_calendar_structure(date_range)
            
            # Merge sources into master calendar
            consolidation_result = self._merge_calendar_sources(
                master_calendar, sources, date_range
            )
            
            # Apply business rules and validations
            self._apply_business_rules(consolidation_result.master_calendar)
            
            # Generate metadata
            metadata = self._generate_consolidation_metadata(sources, consolidation_result)
            consolidation_result.consolidation_metadata = metadata
            
            logger.info(f"Calendar consolidation completed: {len(consolidation_result.master_calendar)} records")
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Calendar consolidation failed: {str(e)}")
            return ConsolidationResult(
                success=False,
                master_calendar=[],
                conflicts_resolved=0,
                records_merged=0,
                sources_processed=0,
                consolidation_metadata={},
                errors=[f"Consolidation failed: {str(e)}"]
            )
    
    def _collect_calendar_sources(
        self,
        executive_config: Optional[ExecutiveCalendarConfig],
        date_range: Tuple[date, date]
    ) -> List[CalendarSource]:
        """Collect data from all calendar sources"""
        sources = []
        
        try:
            # Executive Calendar Source
            if executive_config:
                exec_data = self._extract_executive_calendar_data(executive_config, date_range)
                if exec_data:
                    sources.append(CalendarSource(
                        name="executive_calendar",
                        model_type="ExecutiveCalendarConfig",
                        priority=self.source_priorities['executive_calendar'],
                        last_updated=datetime.now(),
                        record_count=len(exec_data),
                        data=exec_data,
                        metadata={"source": "executive_config", "organization": executive_config.organization_name}
                    ))
            
            # Causal Marketing Calendar Source
            causal_data = self._extract_causal_marketing_data(date_range)
            if causal_data:
                sources.append(CalendarSource(
                    name="causal_marketing",
                    model_type="CausalMarketingCalendar",
                    priority=self.source_priorities['causal_marketing'],
                    last_updated=datetime.now(),
                    record_count=len(causal_data),
                    data=causal_data,
                    metadata={"source": "causal_marketing_model"}
                ))
            
            # Standard Calendar Dimension Source
            calendar_data = self._extract_calendar_dimension_data(date_range)
            if calendar_data:
                sources.append(CalendarSource(
                    name="calendar_dimension",
                    model_type="CalendarDimension",
                    priority=self.source_priorities['calendar_dimension'],
                    last_updated=datetime.now(),
                    record_count=len(calendar_data),
                    data=calendar_data,
                    metadata={"source": "calendar_dimension_model"}
                ))
            
            # Marketing Calendar Source
            marketing_data = self._extract_marketing_calendar_data(date_range)
            if marketing_data:
                sources.append(CalendarSource(
                    name="marketing_calendar",
                    model_type="MarketingCalendar",
                    priority=self.source_priorities['marketing_calendar'],
                    last_updated=datetime.now(),
                    record_count=len(marketing_data),
                    data=marketing_data,
                    metadata={"source": "marketing_model"}
                ))
            
            # Sort sources by priority (highest first)
            sources.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Collected {len(sources)} calendar sources")
            return sources
            
        except Exception as e:
            logger.error(f"Error collecting calendar sources: {str(e)}")
            return []
    
    def _extract_executive_calendar_data(
        self,
        config: ExecutiveCalendarConfig,
        date_range: Tuple[date, date]
    ) -> List[Dict[str, Any]]:
        """Extract data from executive calendar configuration"""
        data = []
        
        try:
            current_date = date_range[0]
            end_date = date_range[1]
            
            while current_date <= end_date:
                # Base calendar record
                record = {
                    'date': current_date,
                    'year': current_date.year,
                    'month': current_date.month,
                    'day': current_date.day,
                    'quarter': (current_date.month - 1) // 3 + 1,
                    'week_of_year': current_date.isocalendar()[1],
                    'day_of_week': current_date.weekday() + 1,
                    'day_name': current_date.strftime('%A'),
                    'month_name': current_date.strftime('%B'),
                    'is_weekend': current_date.weekday() >= 5,
                    'is_holiday': False,
                    'fiscal_year': self._calculate_fiscal_year(current_date, config.fiscal_year_start),
                    'fiscal_quarter': self._calculate_fiscal_quarter(current_date, config.fiscal_year_start),
                    'source': 'executive_calendar',
                    'priority': self.source_priorities['executive_calendar']
                }
                
                # Add business events
                for event in config.custom_events:
                    if event.start_date <= current_date <= (event.end_date or event.start_date):
                        record.update({
                            'has_business_event': True,
                            'business_event_name': event.name,
                            'business_event_type': event.event_type.value,
                            'business_event_impact': event.impact_level.value,
                            'business_event_description': event.description
                        })
                        break
                else:
                    record.update({
                        'has_business_event': False,
                        'business_event_name': None,
                        'business_event_type': None,
                        'business_event_impact': None,
                        'business_event_description': None
                    })
                
                # Add industry-specific attributes
                record.update(self._get_industry_attributes(current_date, config.industry))
                
                data.append(record)
                current_date += timedelta(days=1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting executive calendar data: {str(e)}")
            return []
    
    def _extract_causal_marketing_data(self, date_range: Tuple[date, date]) -> List[Dict[str, Any]]:
        """Extract data from causal marketing calendar model"""
        data = []
        
        try:
            # Query causal marketing calendar data
            query = text("""
                SELECT 
                    date,
                    is_treatment_period,
                    is_control_period,
                    campaign_id,
                    treatment_intensity,
                    expected_lift,
                    confidence_interval_lower,
                    confidence_interval_upper,
                    causal_impact_score
                FROM causal_marketing_calendar 
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date
            """)
            
            result = self.db_session.execute(query, {
                'start_date': date_range[0],
                'end_date': date_range[1]
            })
            
            for row in result:
                record = {
                    'date': row.date,
                    'is_treatment_period': row.is_treatment_period,
                    'is_control_period': row.is_control_period,
                    'campaign_id': row.campaign_id,
                    'treatment_intensity': row.treatment_intensity,
                    'expected_lift': row.expected_lift,
                    'confidence_interval_lower': row.confidence_interval_lower,
                    'confidence_interval_upper': row.confidence_interval_upper,
                    'causal_impact_score': row.causal_impact_score,
                    'source': 'causal_marketing',
                    'priority': self.source_priorities['causal_marketing']
                }
                data.append(record)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting causal marketing data: {str(e)}")
            return []
    
    def _extract_calendar_dimension_data(self, date_range: Tuple[date, date]) -> List[Dict[str, Any]]:
        """Extract data from standard calendar dimension model"""
        data = []
        
        try:
            # Query calendar dimension data
            query = text("""
                SELECT 
                    date,
                    year,
                    month,
                    day,
                    quarter,
                    week_of_year,
                    day_of_week,
                    day_name,
                    month_name,
                    is_weekend,
                    is_holiday,
                    holiday_name,
                    is_business_day,
                    season,
                    fiscal_year,
                    fiscal_quarter,
                    fiscal_month
                FROM calendar_dimension 
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date
            """)
            
            result = self.db_session.execute(query, {
                'start_date': date_range[0],
                'end_date': date_range[1]
            })
            
            for row in result:
                record = {
                    'date': row.date,
                    'year': row.year,
                    'month': row.month,
                    'day': row.day,
                    'quarter': row.quarter,
                    'week_of_year': row.week_of_year,
                    'day_of_week': row.day_of_week,
                    'day_name': row.day_name,
                    'month_name': row.month_name,
                    'is_weekend': row.is_weekend,
                    'is_holiday': row.is_holiday,
                    'holiday_name': row.holiday_name,
                    'is_business_day': row.is_business_day,
                    'season': row.season,
                    'fiscal_year': row.fiscal_year,
                    'fiscal_quarter': row.fiscal_quarter,
                    'fiscal_month': row.fiscal_month,
                    'source': 'calendar_dimension',
                    'priority': self.source_priorities['calendar_dimension']
                }
                data.append(record)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting calendar dimension data: {str(e)}")
            return []
    
    def _extract_marketing_calendar_data(self, date_range: Tuple[date, date]) -> List[Dict[str, Any]]:
        """Extract data from marketing calendar model"""
        data = []
        
        try:
            # Query marketing calendar data
            query = text("""
                SELECT 
                    date,
                    campaign_start,
                    campaign_end,
                    campaign_name,
                    campaign_type,
                    budget_allocated,
                    expected_impressions,
                    target_audience
                FROM marketing_calendar 
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date
            """)
            
            result = self.db_session.execute(query, {
                'start_date': date_range[0],
                'end_date': date_range[1]
            })
            
            for row in result:
                record = {
                    'date': row.date,
                    'campaign_start': row.campaign_start,
                    'campaign_end': row.campaign_end,
                    'campaign_name': row.campaign_name,
                    'campaign_type': row.campaign_type,
                    'budget_allocated': row.budget_allocated,
                    'expected_impressions': row.expected_impressions,
                    'target_audience': row.target_audience,
                    'source': 'marketing_calendar',
                    'priority': self.source_priorities['marketing_calendar']
                }
                data.append(record)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting marketing calendar data: {str(e)}")
            return []
    
    def _generate_master_calendar_structure(self, date_range: Tuple[date, date]) -> List[Dict[str, Any]]:
        """Generate base master calendar structure"""
        master_calendar = []
        
        current_date = date_range[0]
        end_date = date_range[1]
        
        while current_date <= end_date:
            record = {
                'date': current_date,
                'year': current_date.year,
                'month': current_date.month,
                'day': current_date.day,
                'quarter': (current_date.month - 1) // 3 + 1,
                'week_of_year': current_date.isocalendar()[1],
                'day_of_week': current_date.weekday() + 1,
                'day_name': current_date.strftime('%A'),
                'month_name': current_date.strftime('%B'),
                'is_weekend': current_date.weekday() >= 5,
                'is_business_day': current_date.weekday() < 5,
                'consolidated_sources': [],
                'data_quality_score': 0.0,
                'last_updated': datetime.now()
            }
            
            master_calendar.append(record)
            current_date += timedelta(days=1)
        
        return master_calendar
    
    def _merge_calendar_sources(
        self,
        master_calendar: List[Dict[str, Any]],
        sources: List[CalendarSource],
        date_range: Tuple[date, date]
    ) -> ConsolidationResult:
        """Merge all calendar sources into master calendar"""
        conflicts_resolved = 0
        records_merged = 0
        warnings = []
        errors = []
        
        try:
            # Create date index for efficient lookup
            date_index = {record['date']: i for i, record in enumerate(master_calendar)}
            
            # Process each source in priority order
            for source in sources:
                logger.info(f"Merging source: {source.name} (priority: {source.priority})")
                
                for source_record in source.data:
                    record_date = source_record['date']
                    
                    if record_date in date_index:
                        master_idx = date_index[record_date]
                        master_record = master_calendar[master_idx]
                        
                        # Merge record with conflict resolution
                        merge_result = self._merge_record(
                            master_record, source_record, source
                        )
                        
                        conflicts_resolved += merge_result['conflicts_resolved']
                        records_merged += 1
                        
                        # Track source
                        master_record['consolidated_sources'].append({
                            'source': source.name,
                            'priority': source.priority,
                            'last_updated': source.last_updated
                        })
                        
                        # Update data quality score
                        master_record['data_quality_score'] = self._calculate_data_quality_score(
                            master_record
                        )
            
            return ConsolidationResult(
                success=True,
                master_calendar=master_calendar,
                conflicts_resolved=conflicts_resolved,
                records_merged=records_merged,
                sources_processed=len(sources),
                consolidation_metadata={},
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error merging calendar sources: {str(e)}")
            errors.append(f"Merge failed: {str(e)}")
            
            return ConsolidationResult(
                success=False,
                master_calendar=master_calendar,
                conflicts_resolved=conflicts_resolved,
                records_merged=records_merged,
                sources_processed=len(sources),
                consolidation_metadata={},
                warnings=warnings,
                errors=errors
            )
    
    def _merge_record(
        self,
        master_record: Dict[str, Any],
        source_record: Dict[str, Any],
        source: CalendarSource
    ) -> Dict[str, Any]:
        """Merge a single source record into master record"""
        conflicts_resolved = 0
        
        for key, value in source_record.items():
            if key in ['date', 'source', 'priority']:
                continue  # Skip system fields
            
            if key in master_record:
                # Handle conflict
                if master_record[key] != value and master_record[key] is not None:
                    # Resolve conflict based on strategy
                    if self.conflict_resolution == ConflictResolution.HIGHEST_PRIORITY:
                        # Keep existing if it came from higher priority source
                        existing_sources = master_record.get('consolidated_sources', [])
                        if existing_sources:
                            max_existing_priority = max(s['priority'] for s in existing_sources)
                            if source.priority > max_existing_priority:
                                master_record[key] = value
                                conflicts_resolved += 1
                    elif self.conflict_resolution == ConflictResolution.LATEST_WINS:
                        master_record[key] = value
                        conflicts_resolved += 1
                    elif self.conflict_resolution == ConflictResolution.MERGE_ATTRIBUTES:
                        # Try to merge if possible
                        merged_value = self._merge_attribute_values(
                            master_record[key], value, key
                        )
                        if merged_value != master_record[key]:
                            master_record[key] = merged_value
                            conflicts_resolved += 1
            else:
                # New attribute
                master_record[key] = value
        
        return {'conflicts_resolved': conflicts_resolved}
    
    def _merge_attribute_values(self, existing_value: Any, new_value: Any, attribute_name: str) -> Any:
        """Merge two attribute values intelligently"""
        # String concatenation for certain fields
        if attribute_name in ['holiday_name', 'business_event_name', 'campaign_name']:
            if existing_value and new_value:
                return f"{existing_value}; {new_value}"
            return new_value or existing_value
        
        # Boolean OR for flags
        if attribute_name.startswith('is_') or attribute_name.startswith('has_'):
            return bool(existing_value) or bool(new_value)
        
        # Numeric average for scores
        if attribute_name.endswith('_score') or attribute_name.endswith('_intensity'):
            if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                return (existing_value + new_value) / 2
        
        # Default: keep new value
        return new_value
    
    def _calculate_data_quality_score(self, record: Dict[str, Any]) -> float:
        """Calculate data quality score for a record"""
        total_fields = 0
        populated_fields = 0
        
        for key, value in record.items():
            if key not in ['consolidated_sources', 'data_quality_score', 'last_updated']:
                total_fields += 1
                if value is not None and value != '':
                    populated_fields += 1
        
        base_score = populated_fields / total_fields if total_fields > 0 else 0
        
        # Bonus for multiple sources
        source_count = len(record.get('consolidated_sources', []))
        source_bonus = min(source_count * 0.1, 0.3)  # Max 30% bonus
        
        return min(base_score + source_bonus, 1.0)
    
    def _apply_business_rules(self, master_calendar: List[Dict[str, Any]]):
        """Apply business rules and validations to master calendar"""
        for record in master_calendar:
            # Ensure business day logic
            if record.get('is_weekend', False) or record.get('is_holiday', False):
                record['is_business_day'] = False
            else:
                record['is_business_day'] = True
            
            # Validate fiscal year consistency
            if 'fiscal_year' not in record or record['fiscal_year'] is None:
                record['fiscal_year'] = record['year']
            
            # Ensure data consistency
            if record['day_of_week'] != record['date'].weekday() + 1:
                record['day_of_week'] = record['date'].weekday() + 1
                record['day_name'] = record['date'].strftime('%A')
    
    def _calculate_fiscal_year(self, current_date: date, fiscal_start: Any) -> int:
        """Calculate fiscal year based on fiscal year start"""
        fiscal_start_month = {
            'january': 1, 'april': 4, 'july': 7, 'october': 10
        }.get(str(fiscal_start).lower(), 1)
        
        if current_date.month >= fiscal_start_month:
            return current_date.year
        else:
            return current_date.year - 1
    
    def _calculate_fiscal_quarter(self, current_date: date, fiscal_start: Any) -> int:
        """Calculate fiscal quarter based on fiscal year start"""
        fiscal_start_month = {
            'january': 1, 'april': 4, 'july': 7, 'october': 10
        }.get(str(fiscal_start).lower(), 1)
        
        # Adjust month relative to fiscal year start
        adjusted_month = (current_date.month - fiscal_start_month) % 12
        return (adjusted_month // 3) + 1
    
    def _get_industry_attributes(self, current_date: date, industry: Any) -> Dict[str, Any]:
        """Get industry-specific calendar attributes"""
        attributes = {}
        
        industry_str = str(industry).lower()
        
        if industry_str == 'retail':
            # Retail-specific attributes
            attributes.update({
                'is_back_to_school_season': current_date.month in [7, 8],
                'is_holiday_shopping_season': current_date.month in [11, 12],
                'is_spring_season': current_date.month in [3, 4, 5],
                'is_summer_season': current_date.month in [6, 7, 8]
            })
        elif industry_str == 'b2b':
            # B2B-specific attributes
            attributes.update({
                'is_conference_season': current_date.month in [3, 4, 9, 10],
                'is_budget_planning_season': current_date.month in [10, 11, 12],
                'is_summer_slowdown': current_date.month in [7, 8]
            })
        elif industry_str == 'ecommerce':
            # E-commerce specific attributes
            attributes.update({
                'is_cyber_week': self._is_cyber_week(current_date),
                'is_prime_day_season': current_date.month == 7,
                'is_valentine_season': current_date.month == 2
            })
        
        return attributes
    
    def _is_cyber_week(self, current_date: date) -> bool:
        """Check if date is in Cyber Week (week of Black Friday)"""
        # Black Friday is the fourth Thursday of November
        november_first = date(current_date.year, 11, 1)
        first_thursday = november_first + timedelta(days=(3 - november_first.weekday()) % 7)
        black_friday = first_thursday + timedelta(days=21)  # Fourth Thursday
        
        # Cyber Week is the week containing Black Friday
        week_start = black_friday - timedelta(days=black_friday.weekday())
        week_end = week_start + timedelta(days=6)
        
        return week_start <= current_date <= week_end
    
    def _generate_consolidation_metadata(
        self,
        sources: List[CalendarSource],
        result: ConsolidationResult
    ) -> Dict[str, Any]:
        """Generate metadata about the consolidation process"""
        return {
            'consolidation_timestamp': datetime.now().isoformat(),
            'strategy_used': self.consolidation_strategy.value,
            'conflict_resolution': self.conflict_resolution.value,
            'sources_summary': [
                {
                    'name': source.name,
                    'model_type': source.model_type,
                    'priority': source.priority,
                    'record_count': source.record_count,
                    'last_updated': source.last_updated.isoformat()
                }
                for source in sources
            ],
            'quality_metrics': {
                'total_records': len(result.master_calendar),
                'average_quality_score': sum(
                    record.get('data_quality_score', 0) 
                    for record in result.master_calendar
                ) / len(result.master_calendar) if result.master_calendar else 0,
                'conflicts_resolved': result.conflicts_resolved,
                'sources_processed': result.sources_processed
            }
        }
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status and metrics"""
        try:
            # Query latest consolidation metadata
            query = text("""
                SELECT 
                    consolidation_timestamp,
                    strategy_used,
                    total_records,
                    average_quality_score,
                    conflicts_resolved,
                    sources_processed
                FROM calendar_consolidation_log 
                ORDER BY consolidation_timestamp DESC 
                LIMIT 1
            """)
            
            result = self.db_session.execute(query)
            row = result.fetchone()
            
            if row:
                return {
                    'last_consolidation': row.consolidation_timestamp,
                    'strategy': row.strategy_used,
                    'total_records': row.total_records,
                    'quality_score': row.average_quality_score,
                    'conflicts_resolved': row.conflicts_resolved,
                    'sources_processed': row.sources_processed,
                    'status': 'active'
                }
            else:
                return {
                    'status': 'not_initialized',
                    'message': 'No consolidation has been performed yet'
                }
                
        except Exception as e:
            logger.error(f"Error getting consolidation status: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error retrieving status: {str(e)}"
            }
    
    def save_consolidated_calendar(self, result: ConsolidationResult) -> bool:
        """Save consolidated calendar to database"""
        try:
            # Save master calendar records
            for record in result.master_calendar:
                # Convert to database format
                db_record = {
                    'date': record['date'],
                    'year': record['year'],
                    'month': record['month'],
                    'day': record['day'],
                    'quarter': record['quarter'],
                    'week_of_year': record['week_of_year'],
                    'day_of_week': record['day_of_week'],
                    'day_name': record['day_name'],
                    'month_name': record['month_name'],
                    'is_weekend': record['is_weekend'],
                    'is_business_day': record['is_business_day'],
                    'is_holiday': record.get('is_holiday', False),
                    'holiday_name': record.get('holiday_name'),
                    'fiscal_year': record.get('fiscal_year'),
                    'fiscal_quarter': record.get('fiscal_quarter'),
                    'data_quality_score': record['data_quality_score'],
                    'consolidated_sources': str(record['consolidated_sources']),
                    'last_updated': record['last_updated']
                }
                
                # Insert or update record
                insert_query = text("""
                    INSERT INTO master_calendar_dimension
                    (date, year, month, day, quarter, week_of_year, day_of_week,
                     day_name, month_name, is_weekend, is_business_day, is_holiday,
                     holiday_name, fiscal_year, fiscal_quarter, data_quality_score,
                     consolidated_sources, last_updated)
                    VALUES
                    (:date, :year, :month, :day, :quarter, :week_of_year, :day_of_week,
                     :day_name, :month_name, :is_weekend, :is_business_day, :is_holiday,
                     :holiday_name, :fiscal_year, :fiscal_quarter, :data_quality_score,
                     :consolidated_sources, :last_updated)
                    ON CONFLICT (date) DO UPDATE SET
                        year = EXCLUDED.year,
                        month = EXCLUDED.month,
                        day = EXCLUDED.day,
                        quarter = EXCLUDED.quarter,
                        week_of_year = EXCLUDED.week_of_year,
                        day_of_week = EXCLUDED.day_of_week,
                        day_name = EXCLUDED.day_name,
                        month_name = EXCLUDED.month_name,
                        is_weekend = EXCLUDED.is_weekend,
                        is_business_day = EXCLUDED.is_business_day,
                        is_holiday = EXCLUDED.is_holiday,
                        holiday_name = EXCLUDED.holiday_name,
                        fiscal_year = EXCLUDED.fiscal_year,
                        fiscal_quarter = EXCLUDED.fiscal_quarter,
                        data_quality_score = EXCLUDED.data_quality_score,
                        consolidated_sources = EXCLUDED.consolidated_sources,
                        last_updated = EXCLUDED.last_updated
                """)
                
                self.db_session.execute(insert_query, db_record)
            
            # Save consolidation metadata
            metadata_query = text("""
                INSERT INTO calendar_consolidation_log
                (consolidation_timestamp, strategy_used, conflict_resolution,
                 total_records, average_quality_score, conflicts_resolved,
                 sources_processed, consolidation_metadata)
                VALUES
                (:timestamp, :strategy, :conflict_resolution, :total_records,
                 :avg_quality, :conflicts, :sources, :metadata)
            """)
            
            self.db_session.execute(metadata_query, {
                'timestamp': datetime.now(),
                'strategy': self.consolidation_strategy.value,
                'conflict_resolution': self.conflict_resolution.value,
                'total_records': len(result.master_calendar),
                'avg_quality': sum(r.get('data_quality_score', 0) for r in result.master_calendar) / len(result.master_calendar),
                'conflicts': result.conflicts_resolved,
                'sources': result.sources_processed,
                'metadata': str(result.consolidation_metadata)
            })
            
            self.db_session.commit()
            logger.info("Consolidated calendar saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving consolidated calendar: {str(e)}")
            self.db_session.rollback()
            return False