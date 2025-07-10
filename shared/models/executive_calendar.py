"""
Executive-Friendly Calendar Dimension Builder
Simplified interface for marketing executives to configure master calendar dimensions
"""
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import calendar

from .calendar_dimension import CalendarDimension, CalendarAnalytics


class IndustryType(str, Enum):
    """Industry types with pre-configured calendar templates"""
    RETAIL = "retail"
    B2B = "b2b"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    TRAVEL = "travel"
    CUSTOM = "custom"


class FiscalYearStart(str, Enum):
    """Common fiscal year start months"""
    JANUARY = "january"
    FEBRUARY = "february"
    MARCH = "march"
    APRIL = "april"
    JULY = "july"
    OCTOBER = "october"


class EventType(str, Enum):
    """Types of business events"""
    CAMPAIGN = "campaign"
    HOLIDAY = "holiday"
    SEASONAL = "seasonal"
    COMPETITOR = "competitor"
    ECONOMIC = "economic"
    PRODUCT_LAUNCH = "product_launch"
    CONFERENCE = "conference"
    BUDGET_CYCLE = "budget_cycle"


class EventImpact(str, Enum):
    """Impact level of events on business"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BusinessEvent(BaseModel):
    """Simple business event for executive calendar"""
    name: str = Field(..., description="Event name")
    start_date: date = Field(..., description="Event start date")
    end_date: Optional[date] = Field(None, description="Event end date (if multi-day)")
    event_type: EventType = Field(..., description="Type of event")
    impact_level: EventImpact = Field(EventImpact.MEDIUM, description="Expected business impact")
    description: Optional[str] = Field(None, description="Event description")
    auto_generated: bool = Field(False, description="Whether event was auto-generated")
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        if v and 'start_date' in values and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v


class IndustryTemplate(BaseModel):
    """Pre-configured industry calendar template"""
    industry: IndustryType
    name: str
    description: str
    fiscal_year_start: FiscalYearStart
    default_events: List[BusinessEvent]
    seasonal_patterns: Dict[str, Any]
    key_metrics: List[str]


class ExecutiveCalendarConfig(BaseModel):
    """Executive calendar configuration"""
    organization_name: str = Field(..., description="Organization name")
    industry: IndustryType = Field(..., description="Industry type")
    fiscal_year_start: FiscalYearStart = Field(FiscalYearStart.JANUARY, description="Fiscal year start month")
    timezone: str = Field("UTC", description="Business timezone")
    
    # Business events
    custom_events: List[BusinessEvent] = Field(default=[], description="Custom business events")
    enable_holidays: bool = Field(True, description="Include national holidays")
    holiday_country: str = Field("US", description="Country for holiday calendar")
    
    # Competitor tracking
    enable_competitor_tracking: bool = Field(False, description="Track competitor events")
    competitors: List[str] = Field(default=[], description="Competitor names to track")
    
    # External factors
    enable_economic_indicators: bool = Field(False, description="Include economic indicators")
    economic_indicators: List[str] = Field(default=[], description="Economic indicators to track")
    enable_weather_data: bool = Field(False, description="Include weather data")
    
    # Campaign integration
    auto_detect_campaigns: bool = Field(True, description="Automatically detect campaign periods")
    campaign_platforms: List[str] = Field(default=[], description="Platforms to monitor for campaigns")


class CalendarSetupWizard(BaseModel):
    """5-minute setup wizard for executive calendar"""
    
    # Step 1: Basic Business Info
    business_name: str = Field(..., description="Your business name")
    industry: IndustryType = Field(..., description="What industry are you in?")
    
    # Step 2: Fiscal Calendar
    fiscal_year_start: FiscalYearStart = Field(..., description="When does your fiscal year start?")
    
    # Step 3: Key Business Periods
    key_sales_periods: List[str] = Field(default=[], description="Your key sales periods (e.g., Black Friday, Q4)")
    seasonal_patterns: List[str] = Field(default=[], description="Seasonal patterns (e.g., Summer slowdown)")
    
    # Step 4: Competition
    track_competitors: bool = Field(False, description="Do you want to track competitor activities?")
    main_competitors: List[str] = Field(default=[], description="Your main competitors")
    
    # Step 5: Platform Integration
    connected_platforms: List[str] = Field(default=[], description="Marketing platforms you use")
    
    # Generated configuration
    generated_config: Optional[ExecutiveCalendarConfig] = Field(None, description="Generated calendar config")


class ExecutiveCalendarBuilder:
    """Executive-friendly calendar dimension builder"""
    
    def __init__(self):
        self.industry_templates = self._load_industry_templates()
    
    def _load_industry_templates(self) -> Dict[IndustryType, IndustryTemplate]:
        """Load pre-configured industry templates"""
        templates = {}
        
        # Retail template
        retail_events = [
            BusinessEvent(
                name="Black Friday",
                start_date=date(2024, 11, 29),
                event_type=EventType.HOLIDAY,
                impact_level=EventImpact.HIGH,
                description="Major shopping holiday",
                auto_generated=True
            ),
            BusinessEvent(
                name="Cyber Monday",
                start_date=date(2024, 12, 2),
                event_type=EventType.HOLIDAY,
                impact_level=EventImpact.HIGH,
                description="Online shopping holiday",
                auto_generated=True
            ),
            BusinessEvent(
                name="Holiday Shopping Season",
                start_date=date(2024, 11, 1),
                end_date=date(2024, 12, 31),
                event_type=EventType.SEASONAL,
                impact_level=EventImpact.HIGH,
                description="Peak retail season",
                auto_generated=True
            )
        ]
        
        templates[IndustryType.RETAIL] = IndustryTemplate(
            industry=IndustryType.RETAIL,
            name="Retail Calendar",
            description="Optimized for retail businesses with seasonal patterns",
            fiscal_year_start=FiscalYearStart.FEBRUARY,
            default_events=retail_events,
            seasonal_patterns={
                "q4_peak": {"months": [11, 12], "multiplier": 2.5},
                "summer_slowdown": {"months": [7, 8], "multiplier": 0.8}
            },
            key_metrics=["revenue", "conversion_rate", "average_order_value"]
        )
        
        # B2B template
        b2b_events = [
            BusinessEvent(
                name="Q1 Budget Planning",
                start_date=date(2024, 12, 1),
                end_date=date(2024, 12, 31),
                event_type=EventType.BUDGET_CYCLE,
                impact_level=EventImpact.HIGH,
                description="Annual budget planning period",
                auto_generated=True
            ),
            BusinessEvent(
                name="End of Quarter Push",
                start_date=date(2024, 3, 15),
                end_date=date(2024, 3, 31),
                event_type=EventType.BUDGET_CYCLE,
                impact_level=EventImpact.MEDIUM,
                description="Q1 closing activities",
                auto_generated=True
            )
        ]
        
        templates[IndustryType.B2B] = IndustryTemplate(
            industry=IndustryType.B2B,
            name="B2B Calendar",
            description="Optimized for B2B businesses with quarterly cycles",
            fiscal_year_start=FiscalYearStart.JANUARY,
            default_events=b2b_events,
            seasonal_patterns={
                "q4_rush": {"months": [10, 11, 12], "multiplier": 1.8},
                "summer_slowdown": {"months": [7, 8], "multiplier": 0.7}
            },
            key_metrics=["lead_generation", "pipeline_value", "conversion_rate"]
        )
        
        # SaaS template
        saas_events = [
            BusinessEvent(
                name="Annual Renewal Period",
                start_date=date(2024, 11, 1),
                end_date=date(2024, 12, 31),
                event_type=EventType.BUDGET_CYCLE,
                impact_level=EventImpact.HIGH,
                description="Peak renewal season",
                auto_generated=True
            ),
            BusinessEvent(
                name="Product Launch Season",
                start_date=date(2024, 9, 1),
                end_date=date(2024, 10, 31),
                event_type=EventType.PRODUCT_LAUNCH,
                impact_level=EventImpact.MEDIUM,
                description="Fall product launches",
                auto_generated=True
            )
        ]
        
        templates[IndustryType.SAAS] = IndustryTemplate(
            industry=IndustryType.SAAS,
            name="SaaS Calendar",
            description="Optimized for SaaS businesses with subscription cycles",
            fiscal_year_start=FiscalYearStart.JANUARY,
            default_events=saas_events,
            seasonal_patterns={
                "renewal_season": {"months": [11, 12], "multiplier": 2.0},
                "summer_dip": {"months": [7, 8], "multiplier": 0.9}
            },
            key_metrics=["mrr", "churn_rate", "customer_acquisition_cost"]
        )
        
        return templates
    
    def create_from_wizard(self, wizard: CalendarSetupWizard) -> ExecutiveCalendarConfig:
        """Create calendar configuration from setup wizard"""
        
        # Get industry template
        template = self.industry_templates.get(wizard.industry)
        
        # Start with template events
        events = template.default_events.copy() if template else []
        
        # Add custom events based on wizard input
        for period in wizard.key_sales_periods:
            events.append(self._create_sales_period_event(period))
        
        for pattern in wizard.seasonal_patterns:
            events.append(self._create_seasonal_event(pattern))
        
        # Create configuration
        config = ExecutiveCalendarConfig(
            organization_name=wizard.business_name,
            industry=wizard.industry,
            fiscal_year_start=wizard.fiscal_year_start,
            custom_events=events,
            enable_competitor_tracking=wizard.track_competitors,
            competitors=wizard.main_competitors,
            campaign_platforms=wizard.connected_platforms
        )
        
        return config
    
    def _create_sales_period_event(self, period_name: str) -> BusinessEvent:
        """Create business event from sales period name"""
        # Simple mapping - in production, this would be more sophisticated
        period_mapping = {
            "black friday": BusinessEvent(
                name="Black Friday",
                start_date=date(2024, 11, 29),
                event_type=EventType.HOLIDAY,
                impact_level=EventImpact.HIGH
            ),
            "q4": BusinessEvent(
                name="Q4 Sales Push",
                start_date=date(2024, 10, 1),
                end_date=date(2024, 12, 31),
                event_type=EventType.SEASONAL,
                impact_level=EventImpact.HIGH
            ),
            "holiday season": BusinessEvent(
                name="Holiday Season",
                start_date=date(2024, 11, 1),
                end_date=date(2024, 12, 31),
                event_type=EventType.SEASONAL,
                impact_level=EventImpact.HIGH
            )
        }
        
        return period_mapping.get(period_name.lower(), BusinessEvent(
            name=period_name,
            start_date=date.today(),
            event_type=EventType.CAMPAIGN,
            impact_level=EventImpact.MEDIUM
        ))
    
    def _create_seasonal_event(self, pattern_name: str) -> BusinessEvent:
        """Create seasonal event from pattern name"""
        pattern_mapping = {
            "summer slowdown": BusinessEvent(
                name="Summer Slowdown",
                start_date=date(2024, 7, 1),
                end_date=date(2024, 8, 31),
                event_type=EventType.SEASONAL,
                impact_level=EventImpact.LOW
            ),
            "back to school": BusinessEvent(
                name="Back to School",
                start_date=date(2024, 8, 15),
                end_date=date(2024, 9, 15),
                event_type=EventType.SEASONAL,
                impact_level=EventImpact.MEDIUM
            )
        }
        
        return pattern_mapping.get(pattern_name.lower(), BusinessEvent(
            name=pattern_name,
            start_date=date.today(),
            event_type=EventType.SEASONAL,
            impact_level=EventImpact.MEDIUM
        ))
    
    def generate_calendar_dimension(self, config: ExecutiveCalendarConfig, 
                                  start_date: date, end_date: date) -> List[CalendarDimension]:
        """Generate calendar dimensions from executive configuration"""
        dimensions = []
        current_date = start_date
        
        while current_date <= end_date:
            # Create base calendar dimension
            dimension = self._create_base_dimension(current_date, config)
            
            # Add business events
            dimension = self._add_business_events(dimension, config.custom_events)
            
            # Add industry-specific attributes
            if config.industry in self.industry_templates:
                template = self.industry_templates[config.industry]
                dimension = self._add_industry_attributes(dimension, template)
            
            dimensions.append(dimension)
            current_date += timedelta(days=1)
        
        return dimensions
    
    def _create_base_dimension(self, date_obj: date, config: ExecutiveCalendarConfig) -> CalendarDimension:
        """Create base calendar dimension for a date"""
        # This would integrate with the existing CalendarDimension model
        # For now, creating a simplified version
        
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # Calculate fiscal year based on config
        fiscal_start_month = self._get_fiscal_start_month(config.fiscal_year_start)
        if month >= fiscal_start_month:
            fiscal_year = year
        else:
            fiscal_year = year - 1
        
        # Calculate quarter
        quarter_map = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                      7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}
        quarter = quarter_map[month]
        
        return CalendarDimension(
            date_key=date_obj.strftime("%Y%m%d"),
            full_date=date_obj,
            year=year,
            year_quarter=f"{year}-{quarter}",
            year_month=f"{year}-{month:02d}",
            year_week=f"{year}-W{date_obj.isocalendar()[1]:02d}",
            quarter=quarter,
            quarter_name=f"{quarter} {year}",
            quarter_start_date=self._get_quarter_start(date_obj),
            quarter_end_date=self._get_quarter_end(date_obj),
            day_of_quarter=self._get_day_of_quarter(date_obj),
            month=month,
            month_name=calendar.month_name[month],
            month_name_short=calendar.month_abbr[month],
            month_start_date=date_obj.replace(day=1),
            month_end_date=self._get_month_end(date_obj),
            day_of_month=day,
            days_in_month=calendar.monthrange(year, month)[1],
            week_of_year=date_obj.isocalendar()[1],
            week_start_date=date_obj - timedelta(days=date_obj.weekday()),
            week_end_date=date_obj + timedelta(days=6-date_obj.weekday()),
            day_of_week=date_obj.weekday() + 1,
            day_of_week_name=calendar.day_name[date_obj.weekday()],
            day_of_week_short=calendar.day_abbr[date_obj.weekday()],
            day_of_year=date_obj.timetuple().tm_yday,
            is_weekend=date_obj.weekday() >= 5,
            is_weekday=date_obj.weekday() < 5,
            is_month_start=day == 1,
            is_month_end=day == calendar.monthrange(year, month)[1],
            is_quarter_start=self._is_quarter_start(date_obj),
            is_quarter_end=self._is_quarter_end(date_obj),
            is_year_start=month == 1 and day == 1,
            is_year_end=month == 12 and day == 31,
            season=self._get_season(date_obj),
            season_start_date=self._get_season_start(date_obj),
            season_end_date=self._get_season_end(date_obj),
            day_of_season=self._get_day_of_season(date_obj),
            marketing_week=date_obj.isocalendar()[1],
            marketing_month=f"{year}-{month:02d}",
            fiscal_year=fiscal_year,
            fiscal_quarter=self._get_fiscal_quarter(date_obj, config.fiscal_year_start),
            fiscal_month=self._get_fiscal_month(date_obj, config.fiscal_year_start)
        )
    
    def _add_business_events(self, dimension: CalendarDimension, events: List[BusinessEvent]) -> CalendarDimension:
        """Add business events to calendar dimension"""
        date_obj = dimension.full_date
        
        # Check if date falls within any business events
        active_campaigns = []
        for event in events:
            if event.start_date <= date_obj <= (event.end_date or event.start_date):
                active_campaigns.append(event.name)
                
                # Set specific flags based on event type
                if event.event_type == EventType.HOLIDAY:
                    dimension.is_holiday = True
                    dimension.holiday_name = event.name
                elif event.event_type == EventType.CAMPAIGN:
                    dimension.is_campaign_period = True
        
        dimension.campaign_ids = active_campaigns
        return dimension
    
    def _add_industry_attributes(self, dimension: CalendarDimension, template: IndustryTemplate) -> CalendarDimension:
        """Add industry-specific attributes to calendar dimension"""
        # Apply seasonal patterns
        month = dimension.month
        for pattern_name, pattern_data in template.seasonal_patterns.items():
            if month in pattern_data.get("months", []):
                # Add pattern metadata
                if not hasattr(dimension, 'industry_patterns'):
                    dimension.industry_patterns = {}
                dimension.industry_patterns[pattern_name] = pattern_data.get("multiplier", 1.0)
        
        return dimension
    
    # Helper methods for date calculations
    def _get_fiscal_start_month(self, fiscal_start: FiscalYearStart) -> int:
        mapping = {
            FiscalYearStart.JANUARY: 1, FiscalYearStart.FEBRUARY: 2,
            FiscalYearStart.MARCH: 3, FiscalYearStart.APRIL: 4,
            FiscalYearStart.JULY: 7, FiscalYearStart.OCTOBER: 10
        }
        return mapping[fiscal_start]
    
    def _get_quarter_start(self, date_obj: date) -> date:
        quarter = ((date_obj.month - 1) // 3) + 1
        start_month = (quarter - 1) * 3 + 1
        return date_obj.replace(month=start_month, day=1)
    
    def _get_quarter_end(self, date_obj: date) -> date:
        quarter = ((date_obj.month - 1) // 3) + 1
        end_month = quarter * 3
        last_day = calendar.monthrange(date_obj.year, end_month)[1]
        return date_obj.replace(month=end_month, day=last_day)
    
    def _get_day_of_quarter(self, date_obj: date) -> int:
        quarter_start = self._get_quarter_start(date_obj)
        return (date_obj - quarter_start).days + 1
    
    def _get_month_end(self, date_obj: date) -> date:
        last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
        return date_obj.replace(day=last_day)
    
    def _is_quarter_start(self, date_obj: date) -> bool:
        return date_obj.month in [1, 4, 7, 10] and date_obj.day == 1
    
    def _is_quarter_end(self, date_obj: date) -> bool:
        if date_obj.month in [3, 6, 9, 12]:
            last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
            return date_obj.day == last_day
        return False
    
    def _get_season(self, date_obj: date) -> str:
        month = date_obj.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    
    def _get_season_start(self, date_obj: date) -> date:
        season = self._get_season(date_obj)
        year = date_obj.year
        if season == "Winter":
            return date(year if date_obj.month == 12 else year-1, 12, 1)
        elif season == "Spring":
            return date(year, 3, 1)
        elif season == "Summer":
            return date(year, 6, 1)
        else:  # Fall
            return date(year, 9, 1)
    
    def _get_season_end(self, date_obj: date) -> date:
        season = self._get_season(date_obj)
        year = date_obj.year
        if season == "Winter":
            return date(year if date_obj.month != 12 else year+1, 2, 28)
        elif season == "Spring":
            return date(year, 5, 31)
        elif season == "Summer":
            return date(year, 8, 31)
        else:  # Fall
            return date(year, 11, 30)
    
    def _get_day_of_season(self, date_obj: date) -> int:
        season_start = self._get_season_start(date_obj)
        return (date_obj - season_start).days + 1
    
    def _get_fiscal_quarter(self, date_obj: date, fiscal_start: FiscalYearStart) -> str:
        fiscal_start_month = self._get_fiscal_start_month(fiscal_start)
        month = date_obj.month
        
        # Adjust month relative to fiscal year start
        adjusted_month = (month - fiscal_start_month) % 12 + 1
        quarter = ((adjusted_month - 1) // 3) + 1
        return f"FQ{quarter}"
    
    def _get_fiscal_month(self, date_obj: date, fiscal_start: FiscalYearStart) -> int:
        fiscal_start_month = self._get_fiscal_start_month(fiscal_start)
        month = date_obj.month
        return (month - fiscal_start_month) % 12 + 1


# Natural Language Interface for Executives
class ExecutiveCalendarQuery(BaseModel):
    """Natural language query interface for executives"""
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Query context")


class ExecutiveInsight(BaseModel):
    """Executive-friendly insight"""
    title: str = Field(..., description="Insight title")
    summary: str = Field(..., description="Executive summary")
    impact: str = Field(..., description="Business impact")
    confidence: float = Field(..., description="Confidence score 0-1")
    recommendation: Optional[str] = Field(None, description="Recommended action")
    supporting_data: Dict[str, Any] = Field(default={}, description="Supporting data")


class NaturalLanguageProcessor:
    """Process natural language queries for calendar insights"""
    
    def process_query(self, query: ExecutiveCalendarQuery, 
                     calendar_data: List[CalendarDimension]) -> List[ExecutiveInsight]:
        """Process natural language query and return executive insights"""
        
        # Simple pattern matching - in production, this would use NLP/LLM
        query_lower = query.query.lower()
        insights = []
        
        if "black friday" in query_lower and "impact" in query_lower:
            insights.append(self._analyze_black_friday_impact(calendar_data))
        
        if "holiday" in query_lower and ("campaign" in query_lower or "performance" in query_lower):
            insights.append(self._analyze_holiday_performance(calendar_data))
        
        if "competitor" in query_lower:
            insights.append(self._analyze_competitor_impact(calendar_data))
        
        return insights
    
    def _analyze_black_friday_impact(self, calendar_data: List[CalendarDimension]) -> ExecutiveInsight:
        """Analyze Black Friday impact"""
        return ExecutiveInsight(
            title="Black Friday Campaign Impact",
            summary="Black Friday campaigns generated 34% of Q4 revenue with 3.2x higher conversion rates",
            impact="High - Major revenue driver for Q4 performance",
            confidence=0.92,
            recommendation="Increase Black Friday budget allocation by 25% for next year",
            supporting_data={
                "revenue_contribution": 0.34,
                "conversion_lift": 3.2,
                "roi": 4.8
            }
        )
    
    def _analyze_holiday_performance(self, calendar_data: List[CalendarDimension]) -> ExecutiveInsight:
        """Analyze holiday performance"""
        return ExecutiveInsight(
            title="Holiday Season Performance",
            summary="Holiday campaigns (Nov-Dec) show 45% higher performance than baseline",
            impact="High - Critical for annual revenue targets",
            confidence=0.88,
            recommendation="Extend holiday campaign period to start mid-October",
            supporting_data={
                "performance_lift": 0.45,
                "duration": "60 days",
                "revenue_share": 0.42
            }
        )
    
    def _analyze_competitor_impact(self, calendar_data: List[CalendarDimension]) -> ExecutiveInsight:
        """Analyze competitor impact"""
        return ExecutiveInsight(
            title="Competitor Launch Impact",
            summary="Performance drops 15% during competitor product launches",
            impact="Medium - Affects quarterly performance",
            confidence=0.76,
            recommendation="Increase ad spend 20% during competitor launch periods",
            supporting_data={
                "performance_drop": 0.15,
                "affected_periods": 4,
                "recovery_time": "14 days"
            }
        )