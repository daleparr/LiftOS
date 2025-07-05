"""
Calendar Dimension Generator for LiftOS
Generates comprehensive calendar dimension data for temporal analysis
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import calendar
from shared.models.calendar_dimension import (
    CalendarDimension, DayOfWeek, Month, Quarter, Season
)


class CalendarDimensionGenerator:
    """Generator for calendar dimension data"""
    
    def __init__(self):
        self.holidays = self._get_us_holidays()
        self.marketing_events = self._get_marketing_events()
    
    def generate_calendar_data(
        self, 
        start_date: date, 
        end_date: date,
        include_holidays: bool = True,
        include_marketing_calendar: bool = True
    ) -> List[CalendarDimension]:
        """Generate calendar dimension data for date range"""
        
        calendar_data = []
        current_date = start_date
        
        while current_date <= end_date:
            calendar_entry = self._create_calendar_entry(
                current_date, 
                include_holidays, 
                include_marketing_calendar
            )
            calendar_data.append(calendar_entry)
            current_date += timedelta(days=1)
        
        return calendar_data
    
    def _create_calendar_entry(
        self, 
        target_date: date, 
        include_holidays: bool, 
        include_marketing_calendar: bool
    ) -> CalendarDimension:
        """Create a single calendar dimension entry"""
        
        # Basic date attributes
        year = target_date.year
        month = target_date.month
        day = target_date.day
        
        # Date key
        date_key = target_date.strftime("%Y%m%d")
        
        # Quarter calculations
        quarter_num = (month - 1) // 3 + 1
        quarter = Quarter(f"Q{quarter_num}")
        quarter_start = date(year, (quarter_num - 1) * 3 + 1, 1)
        quarter_end = date(year, quarter_num * 3, calendar.monthrange(year, quarter_num * 3)[1])
        
        # Month calculations
        month_name = Month(calendar.month_name[month])
        month_start = date(year, month, 1)
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        
        # Week calculations
        iso_year, iso_week, iso_weekday = target_date.isocalendar()
        week_start = target_date - timedelta(days=iso_weekday - 1)
        week_end = week_start + timedelta(days=6)
        
        # Day of week
        weekday = target_date.weekday()  # 0=Monday, 6=Sunday
        day_names = [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY, 
                    DayOfWeek.THURSDAY, DayOfWeek.FRIDAY, DayOfWeek.SATURDAY, DayOfWeek.SUNDAY]
        day_name = day_names[weekday]
        
        # Season calculation
        season = self._get_season(target_date)
        season_dates = self._get_season_dates(year, season)
        
        # Holiday information
        holiday_info = self._get_holiday_info(target_date) if include_holidays else {}
        
        # Marketing calendar
        marketing_info = self._get_marketing_info(target_date) if include_marketing_calendar else {}
        
        # Business flags
        is_weekend = weekday >= 5
        is_weekday = not is_weekend
        
        return CalendarDimension(
            date_key=date_key,
            full_date=target_date,
            
            # Year attributes
            year=year,
            year_quarter=f"{year}-{quarter.value}",
            year_month=f"{year}-{month:02d}",
            year_week=f"{year}-W{iso_week:02d}",
            
            # Quarter attributes
            quarter=quarter,
            quarter_name=f"{quarter.value} {year}",
            quarter_start_date=quarter_start,
            quarter_end_date=quarter_end,
            day_of_quarter=(target_date - quarter_start).days + 1,
            
            # Month attributes
            month=month,
            month_name=month_name,
            month_name_short=calendar.month_abbr[month],
            month_start_date=month_start,
            month_end_date=month_end,
            day_of_month=day,
            days_in_month=calendar.monthrange(year, month)[1],
            
            # Week attributes
            week_of_year=iso_week,
            week_start_date=week_start,
            week_end_date=week_end,
            day_of_week=weekday + 1,  # 1=Monday, 7=Sunday
            day_of_week_name=day_name,
            day_of_week_short=day_name.value[:3],
            
            # Day attributes
            day_of_year=target_date.timetuple().tm_yday,
            
            # Business calendar
            is_weekend=is_weekend,
            is_weekday=is_weekday,
            is_month_start=(day == 1),
            is_month_end=(target_date == month_end),
            is_quarter_start=(target_date == quarter_start),
            is_quarter_end=(target_date == quarter_end),
            is_year_start=(month == 1 and day == 1),
            is_year_end=(month == 12 and day == 31),
            
            # Holiday information
            is_holiday=holiday_info.get('is_holiday', False),
            holiday_name=holiday_info.get('holiday_name'),
            is_black_friday=holiday_info.get('is_black_friday', False),
            is_cyber_monday=holiday_info.get('is_cyber_monday', False),
            is_prime_day=holiday_info.get('is_prime_day', False),
            
            # Seasonal attributes
            season=season,
            season_start_date=season_dates['start'],
            season_end_date=season_dates['end'],
            day_of_season=(target_date - season_dates['start']).days + 1,
            
            # Marketing calendar
            marketing_week=marketing_info.get('marketing_week', iso_week),
            marketing_month=marketing_info.get('marketing_month', f"{year}-{month:02d}"),
            fiscal_year=marketing_info.get('fiscal_year', year),
            fiscal_quarter=marketing_info.get('fiscal_quarter', quarter.value),
            fiscal_month=marketing_info.get('fiscal_month', month),
            
            # Relative dates (calculated from today)
            days_from_today=(target_date - date.today()).days,
            weeks_from_today=((target_date - date.today()).days // 7),
            months_from_today=self._months_between(date.today(), target_date),
            
            # Causal analysis attributes (defaults)
            is_campaign_period=False,
            campaign_ids=[],
            is_treatment_period=False,
            is_control_period=False,
            
            # External factors (empty defaults)
            economic_indicators={},
            weather_data={},
            competitor_events=[]
        )
    
    def _get_season(self, target_date: date) -> Season:
        """Determine meteorological season"""
        month = target_date.month
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL
    
    def _get_season_dates(self, year: int, season: Season) -> Dict[str, date]:
        """Get start and end dates for a season"""
        if season == Season.WINTER:
            # Winter spans two years
            if date.today().month >= 12:
                return {
                    'start': date(year, 12, 1),
                    'end': date(year + 1, 2, 28 if not calendar.isleap(year + 1) else 29)
                }
            else:
                return {
                    'start': date(year - 1, 12, 1),
                    'end': date(year, 2, 28 if not calendar.isleap(year) else 29)
                }
        elif season == Season.SPRING:
            return {'start': date(year, 3, 1), 'end': date(year, 5, 31)}
        elif season == Season.SUMMER:
            return {'start': date(year, 6, 1), 'end': date(year, 8, 31)}
        else:  # FALL
            return {'start': date(year, 9, 1), 'end': date(year, 11, 30)}
    
    def _get_holiday_info(self, target_date: date) -> Dict[str, Any]:
        """Get holiday information for a date"""
        holiday_info = {
            'is_holiday': False,
            'holiday_name': None,
            'is_black_friday': False,
            'is_cyber_monday': False,
            'is_prime_day': False
        }
        
        # Check for holidays
        date_key = target_date.strftime("%m-%d")
        year = target_date.year
        
        # Fixed holidays
        if date_key in self.holidays:
            holiday_info['is_holiday'] = True
            holiday_info['holiday_name'] = self.holidays[date_key]
        
        # Black Friday (4th Thursday of November + 1 day)
        thanksgiving = self._get_nth_weekday(year, 11, 3, 4)  # 4th Thursday
        black_friday = thanksgiving + timedelta(days=1)
        if target_date == black_friday:
            holiday_info['is_black_friday'] = True
            holiday_info['holiday_name'] = "Black Friday"
        
        # Cyber Monday (Monday after Black Friday)
        cyber_monday = black_friday + timedelta(days=3)
        if target_date == cyber_monday:
            holiday_info['is_cyber_monday'] = True
            holiday_info['holiday_name'] = "Cyber Monday"
        
        # Amazon Prime Day (typically mid-July, varies by year)
        if target_date.month == 7 and 10 <= target_date.day <= 16:
            holiday_info['is_prime_day'] = True
            holiday_info['holiday_name'] = "Amazon Prime Day"
        
        return holiday_info
    
    def _get_marketing_info(self, target_date: date) -> Dict[str, Any]:
        """Get marketing calendar information"""
        year = target_date.year
        
        # Marketing week (Sunday start)
        days_since_sunday = (target_date.weekday() + 1) % 7
        week_start = target_date - timedelta(days=days_since_sunday)
        marketing_week = week_start.isocalendar()[1]
        
        return {
            'marketing_week': marketing_week,
            'marketing_month': f"{year}-{target_date.month:02d}",
            'fiscal_year': year if target_date.month >= 1 else year - 1,
            'fiscal_quarter': f"Q{(target_date.month - 1) // 3 + 1}",
            'fiscal_month': target_date.month
        }
    
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth occurrence of a weekday in a month"""
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()
        
        # Calculate days to add to get to the first occurrence of the target weekday
        days_to_add = (weekday - first_weekday) % 7
        first_occurrence = first_day + timedelta(days=days_to_add)
        
        # Add weeks to get to the nth occurrence
        nth_occurrence = first_occurrence + timedelta(weeks=n-1)
        
        return nth_occurrence
    
    def _months_between(self, start_date: date, end_date: date) -> int:
        """Calculate months between two dates"""
        return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    
    def _get_us_holidays(self) -> Dict[str, str]:
        """Get US federal holidays (fixed dates)"""
        return {
            "01-01": "New Year's Day",
            "07-04": "Independence Day",
            "12-25": "Christmas Day",
            "11-11": "Veterans Day",
            "02-14": "Valentine's Day",
            "03-17": "St. Patrick's Day",
            "10-31": "Halloween",
            "12-24": "Christmas Eve",
            "12-31": "New Year's Eve"
        }
    
    def _get_marketing_events(self) -> Dict[str, str]:
        """Get marketing events and shopping seasons"""
        return {
            "back_to_school": "August-September",
            "holiday_season": "November-December",
            "spring_cleaning": "March-April",
            "summer_vacation": "June-August"
        }

    def generate_sample_data(self, num_days: int = 20) -> List[CalendarDimension]:
        """Generate sample calendar data for EDA"""
        end_date = date.today()
        start_date = end_date - timedelta(days=num_days - 1)
        
        return self.generate_calendar_data(start_date, end_date)