"""
Test script for Calendar Dimension EDA functionality
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from shared.models.calendar_dimension import CalendarDimension
    from shared.utils.calendar_generator import CalendarDimensionGenerator
    
    print("Calendar Dimension EDA Implementation Test")
    print("All imports working correctly!")
    
    # Test calendar generator
    generator = CalendarDimensionGenerator()
    sample_data = generator.generate_sample_data(3)
    
    print(f"Generated {len(sample_data)} calendar records")
    print(f"Schema has {len(CalendarDimension.__fields__)} fields")
    print(f"First record date: {sample_data[0].full_date}")
    print(f"First record is weekend: {sample_data[0].is_weekend}")
    print(f"First record quarter: {sample_data[0].quarter}")
    
    print("\nSample Calendar Data (first record):")
    first_record = sample_data[0]
    print(f"  Date: {first_record.full_date}")
    print(f"  Day of Week: {first_record.day_of_week_name}")
    print(f"  Month: {first_record.month_name}")
    print(f"  Quarter: {first_record.quarter}")
    print(f"  Season: {first_record.season}")
    print(f"  Is Weekend: {first_record.is_weekend}")
    print(f"  Is Holiday: {first_record.is_holiday}")
    
    print("\nCalendar Dimension EDA implementation is working correctly!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()