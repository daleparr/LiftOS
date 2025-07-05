"""
Calendar Dimension Models for Temporal Analysis in LiftOS
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DayOfWeek(str, Enum):
    """Day of week enumeration"""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


class Month(str, Enum):
    """Month enumeration"""
    JANUARY = "January"
    FEBRUARY = "February"
    MARCH = "March"
    APRIL = "April"
    MAY = "May"
    JUNE = "June"
    JULY = "July"
    AUGUST = "August"
    SEPTEMBER = "September"
    OCTOBER = "October"
    NOVEMBER = "November"
    DECEMBER = "December"


class Quarter(str, Enum):
    """Quarter enumeration"""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class Season(str, Enum):
    """Season enumeration"""
    SPRING = "Spring"
    SUMMER = "Summer"
    FALL = "Fall"
    WINTER = "Winter"


class CalendarDimension(BaseModel):
    """
    Calendar dimension model for temporal analysis in marketing attribution.
    Provides comprehensive date attributes for time-based causal analysis.
    """
    
    # Primary date fields
    date_key: str = Field(..., description="Primary key in YYYYMMDD format")
    full_date: date = Field(..., description="Full date")
    
    # Year attributes
    year: int = Field(..., description="Year (e.g., 2024)")
    year_quarter: str = Field(..., description="Year and quarter (e.g., '2024-Q1')")
    year_month: str = Field(..., description="Year and month (e.g., '2024-01')")
    year_week: str = Field(..., description="Year and ISO week (e.g., '2024-W01')")
    
    # Quarter attributes
    quarter: Quarter = Field(..., description="Quarter of the year")
    quarter_name: str = Field(..., description="Quarter name (e.g., 'Q1 2024')")
    quarter_start_date: date = Field(..., description="First day of the quarter")
    quarter_end_date: date = Field(..., description="Last day of the quarter")
    day_of_quarter: int = Field(..., description="Day number within the quarter (1-92)")
    
    # Month attributes
    month: int = Field(..., description="Month number (1-12)")
    month_name: Month = Field(..., description="Month name")
    month_name_short: str = Field(..., description="Abbreviated month name (e.g., 'Jan')")
    month_start_date: date = Field(..., description="First day of the month")
    month_end_date: date = Field(..., description="Last day of the month")
    day_of_month: int = Field(..., description="Day number within the month (1-31)")
    days_in_month: int = Field(..., description="Total days in the month")
    
    # Week attributes
    week_of_year: int = Field(..., description="ISO week number (1-53)")
    week_start_date: date = Field(..., description="Monday of the week")
    week_end_date: date = Field(..., description="Sunday of the week")
    day_of_week: int = Field(..., description="Day of week number (1=Monday, 7=Sunday)")
    day_of_week_name: DayOfWeek = Field(..., description="Day of week name")
    day_of_week_short: str = Field(..., description="Abbreviated day name (e.g., 'Mon')")
    
    # Day attributes
    day_of_year: int = Field(..., description="Day number within the year (1-366)")
    
    # Business calendar attributes
    is_weekend: bool = Field(..., description="True if Saturday or Sunday")
    is_weekday: bool = Field(..., description="True if Monday through Friday")
    is_month_start: bool = Field(..., description="True if first day of month")
    is_month_end: bool = Field(..., description="True if last day of month")
    is_quarter_start: bool = Field(..., description="True if first day of quarter")
    is_quarter_end: bool = Field(..., description="True if last day of quarter")
    is_year_start: bool = Field(..., description="True if January 1st")
    is_year_end: bool = Field(..., description="True if December 31st")
    
    # Holiday and special events
    is_holiday: bool = Field(default=False, description="True if a recognized holiday")
    holiday_name: Optional[str] = Field(None, description="Name of the holiday if applicable")
    is_black_friday: bool = Field(default=False, description="True if Black Friday")
    is_cyber_monday: bool = Field(default=False, description="True if Cyber Monday")
    is_prime_day: bool = Field(default=False, description="True if Amazon Prime Day")
    
    # Seasonal attributes
    season: Season = Field(..., description="Meteorological season")
    season_start_date: date = Field(..., description="First day of the season")
    season_end_date: date = Field(..., description="Last day of the season")
    day_of_season: int = Field(..., description="Day number within the season")
    
    # Marketing calendar attributes
    marketing_week: int = Field(..., description="Marketing week number (Sunday start)")
    marketing_month: str = Field(..., description="Marketing month (4-4-5 calendar)")
    fiscal_year: int = Field(..., description="Fiscal year")
    fiscal_quarter: str = Field(..., description="Fiscal quarter")
    fiscal_month: int = Field(..., description="Fiscal month")
    
    # Relative date attributes
    days_from_today: Optional[int] = Field(None, description="Days from current date (negative for past)")
    weeks_from_today: Optional[int] = Field(None, description="Weeks from current date")
    months_from_today: Optional[int] = Field(None, description="Months from current date")
    
    # Causal analysis attributes
    is_campaign_period: bool = Field(default=False, description="True if within a campaign period")
    campaign_ids: List[str] = Field(default=[], description="Active campaign IDs for this date")
    is_treatment_period: bool = Field(default=False, description="True if within treatment period")
    is_control_period: bool = Field(default=False, description="True if within control period")
    
    # External factors
    economic_indicators: Dict[str, float] = Field(default={}, description="Economic indicators for the date")
    weather_data: Dict[str, Any] = Field(default={}, description="Weather data if available")
    competitor_events: List[str] = Field(default=[], description="Known competitor events")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    data_source: str = Field(default="system_generated", description="Source of calendar data")
    version: str = Field(default="1.0", description="Calendar dimension version")


class CalendarDimensionRequest(BaseModel):
    """Request model for calendar dimension operations"""
    start_date: date = Field(..., description="Start date for calendar generation")
    end_date: date = Field(..., description="End date for calendar generation")
    include_holidays: bool = Field(default=True, description="Include holiday information")
    include_marketing_calendar: bool = Field(default=True, description="Include marketing calendar attributes")
    include_external_factors: bool = Field(default=False, description="Include external factor data")
    fiscal_year_start_month: int = Field(default=1, description="Fiscal year start month (1-12)")


class CalendarDimensionResponse(BaseModel):
    """Response model for calendar dimension operations"""
    calendar_data: List[CalendarDimension]
    total_days: int
    date_range: Dict[str, str]
    metadata: Dict[str, Any]


class CalendarAnalytics(BaseModel):
    """Analytics model for calendar dimension"""
    total_days: int
    weekdays: int
    weekends: int
    holidays: int
    quarters: Dict[str, int]
    months: Dict[str, int]
    seasons: Dict[str, int]
    campaign_days: int
    treatment_days: int
    control_days: int