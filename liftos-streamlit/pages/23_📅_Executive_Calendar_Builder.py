"""
Executive Calendar Builder
Visual, no-code calendar dimension builder for marketing executives
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date, timedelta
import calendar
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Executive Calendar Builder - LiftOS",
    page_icon="📅",
    layout="wide"
)

# Import shared utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.executive_calendar import (
    IndustryType, FiscalYearStart, EventType, EventImpact,
    BusinessEvent, CalendarSetupWizard, ExecutiveCalendarConfig,
    ExecutiveCalendarBuilder, ExecutiveCalendarQuery, NaturalLanguageProcessor
)
from shared.utils.config import get_service_config
# Authentication imports removed for demo mode

# Service configuration
try:
    config = get_service_config("streamlit", 8501)
    DATA_INGESTION_URL = config.get("data_ingestion_service_url", "http://localhost:8001")
except:
    # Fallback configuration for demo mode
    config = {}
    DATA_INGESTION_URL = "http://localhost:8001"

def initialize_session_state():
    """Initialize session state variables"""
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {}
    if 'calendar_config' not in st.session_state:
        st.session_state.calendar_config = None
    if 'custom_events' not in st.session_state:
        st.session_state.custom_events = []
    if 'calendar_builder' not in st.session_state:
        st.session_state.calendar_builder = ExecutiveCalendarBuilder()

def render_setup_wizard():
    """Render the 5-minute setup wizard"""
    st.header("🚀 5-Minute Calendar Setup Wizard")
    st.markdown("Let's set up your marketing calendar in just a few simple steps!")
    
    # Progress bar
    progress = (st.session_state.wizard_step - 1) / 5
    st.progress(progress)
    st.write(f"Step {st.session_state.wizard_step} of 5")
    
    if st.session_state.wizard_step == 1:
        render_step_1_business_info()
    elif st.session_state.wizard_step == 2:
        render_step_2_fiscal_calendar()
    elif st.session_state.wizard_step == 3:
        render_step_3_business_periods()
    elif st.session_state.wizard_step == 4:
        render_step_4_competition()
    elif st.session_state.wizard_step == 5:
        render_step_5_platforms()

def render_step_1_business_info():
    """Step 1: Basic Business Information"""
    st.subheader("📊 Tell us about your business")
    
    col1, col2 = st.columns(2)
    
    with col1:
        business_name = st.text_input(
            "What's your business name?",
            value=st.session_state.wizard_data.get('business_name', ''),
            placeholder="e.g., Acme Marketing Co."
        )
    
    with col2:
        industry = st.selectbox(
            "What industry are you in?",
            options=[industry.value for industry in IndustryType],
            index=0 if 'industry' not in st.session_state.wizard_data else 
                  list(IndustryType).index(IndustryType(st.session_state.wizard_data['industry'])),
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    # Industry description
    industry_descriptions = {
        'retail': "Physical and online retail businesses with seasonal patterns",
        'b2b': "Business-to-business companies with quarterly cycles",
        'ecommerce': "Online commerce with digital marketing focus",
        'saas': "Software-as-a-Service with subscription models",
        'financial_services': "Banking, insurance, and financial products",
        'healthcare': "Healthcare providers and medical services",
        'education': "Educational institutions and e-learning",
        'travel': "Travel, hospitality, and tourism",
        'custom': "Custom configuration for unique business models"
    }
    
    if industry in industry_descriptions:
        st.info(f"**{industry.replace('_', ' ').title()}**: {industry_descriptions[industry]}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Next →", disabled=not business_name):
            st.session_state.wizard_data.update({
                'business_name': business_name,
                'industry': industry
            })
            st.session_state.wizard_step = 2
            st.rerun()

def render_step_2_fiscal_calendar():
    """Step 2: Fiscal Calendar Setup"""
    st.subheader("📅 When does your fiscal year start?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fiscal_start = st.selectbox(
            "Fiscal Year Start Month",
            options=[start.value for start in FiscalYearStart],
            index=0 if 'fiscal_year_start' not in st.session_state.wizard_data else 
                  list(FiscalYearStart).index(FiscalYearStart(st.session_state.wizard_data['fiscal_year_start'])),
            format_func=lambda x: x.title()
        )
    
    with col2:
        # Show fiscal year example
        current_year = datetime.now().year
        if fiscal_start == 'january':
            fy_example = f"FY{current_year}: Jan {current_year} - Dec {current_year}"
        elif fiscal_start == 'april':
            fy_example = f"FY{current_year}: Apr {current_year} - Mar {current_year + 1}"
        elif fiscal_start == 'july':
            fy_example = f"FY{current_year}: Jul {current_year} - Jun {current_year + 1}"
        elif fiscal_start == 'october':
            fy_example = f"FY{current_year}: Oct {current_year} - Sep {current_year + 1}"
        else:
            fy_example = f"Custom fiscal year starting in {fiscal_start.title()}"
        
        st.info(f"**Example**: {fy_example}")
    
    # Common fiscal year patterns
    st.markdown("**Common patterns:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🏢 **Corporate**: January")
    with col2:
        st.write("🏛️ **Government**: October")
    with col3:
        st.write("🎓 **Education**: July")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.wizard_step = 1
            st.rerun()
    with col3:
        if st.button("Next →"):
            st.session_state.wizard_data['fiscal_year_start'] = fiscal_start
            st.session_state.wizard_step = 3
            st.rerun()

def render_step_3_business_periods():
    """Step 3: Key Business Periods"""
    st.subheader("🎯 What are your key business periods?")
    
    # Pre-defined options based on industry
    industry = st.session_state.wizard_data.get('industry', 'custom')
    
    if industry == 'retail':
        default_periods = ['Black Friday', 'Holiday Season', 'Back to School', 'Spring Sales']
    elif industry == 'b2b':
        default_periods = ['Q4 Budget Planning', 'End of Quarter Push', 'Conference Season']
    elif industry == 'saas':
        default_periods = ['Renewal Season', 'Product Launch Season', 'Conference Season']
    elif industry == 'ecommerce':
        default_periods = ['Black Friday', 'Cyber Monday', 'Prime Day', 'Holiday Shopping']
    else:
        default_periods = ['Q1 Planning', 'Summer Campaign', 'Holiday Season', 'Year-end Push']
    
    st.markdown("**Select your key sales/marketing periods:**")
    
    selected_periods = []
    cols = st.columns(2)
    
    for i, period in enumerate(default_periods):
        with cols[i % 2]:
            if st.checkbox(period, key=f"period_{i}"):
                selected_periods.append(period)
    
    # Custom periods
    st.markdown("**Add custom periods:**")
    custom_period = st.text_input("Custom business period", placeholder="e.g., Annual Conference")
    if custom_period and st.button("Add Custom Period"):
        selected_periods.append(custom_period)
    
    # Seasonal patterns
    st.markdown("**Seasonal patterns:**")
    seasonal_patterns = []
    
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Summer Slowdown"):
            seasonal_patterns.append("Summer Slowdown")
        if st.checkbox("Holiday Rush"):
            seasonal_patterns.append("Holiday Rush")
    
    with col2:
        if st.checkbox("Back to School Boost"):
            seasonal_patterns.append("Back to School Boost")
        if st.checkbox("New Year Planning"):
            seasonal_patterns.append("New Year Planning")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.wizard_step = 2
            st.rerun()
    with col3:
        if st.button("Next →"):
            st.session_state.wizard_data.update({
                'key_sales_periods': selected_periods,
                'seasonal_patterns': seasonal_patterns
            })
            st.session_state.wizard_step = 4
            st.rerun()

def render_step_4_competition():
    """Step 4: Competition Tracking"""
    st.subheader("🏆 Do you want to track competitor activities?")
    
    track_competitors = st.radio(
        "Enable competitor tracking?",
        options=["Yes, track my competitors", "No, skip this step"],
        index=0 if st.session_state.wizard_data.get('track_competitors', False) else 1
    )
    
    competitors = []
    if track_competitors == "Yes, track my competitors":
        st.markdown("**Add your main competitors:**")
        
        # Dynamic competitor input
        if 'competitors' not in st.session_state:
            st.session_state.competitors = ['']
        
        for i, competitor in enumerate(st.session_state.competitors):
            col1, col2 = st.columns([4, 1])
            with col1:
                competitor_name = st.text_input(
                    f"Competitor {i+1}",
                    value=competitor,
                    key=f"competitor_{i}",
                    placeholder="e.g., Competitor Inc."
                )
                if competitor_name:
                    st.session_state.competitors[i] = competitor_name
                    if competitor_name not in competitors:
                        competitors.append(competitor_name)
            
            with col2:
                if len(st.session_state.competitors) > 1:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.competitors.pop(i)
                        st.rerun()
        
        if st.button("+ Add Another Competitor"):
            st.session_state.competitors.append('')
            st.rerun()
        
        st.info("💡 **Tip**: We'll automatically track competitor product launches, major campaigns, and announcements that might impact your performance.")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.wizard_step = 3
            st.rerun()
    with col3:
        if st.button("Next →"):
            st.session_state.wizard_data.update({
                'track_competitors': track_competitors == "Yes, track my competitors",
                'main_competitors': [c for c in competitors if c.strip()]
            })
            st.session_state.wizard_step = 5
            st.rerun()

def render_step_5_platforms():
    """Step 5: Platform Integration"""
    st.subheader("🔗 Which marketing platforms do you use?")
    
    platforms = {
        'Meta Business': 'meta_business',
        'Google Ads': 'google_ads',
        'Klaviyo': 'klaviyo',
        'Shopify': 'shopify',
        'HubSpot': 'hubspot',
        'LinkedIn Ads': 'linkedin_ads',
        'TikTok Ads': 'tiktok',
        'X (Twitter) Ads': 'x_ads',
        'Amazon Seller Central': 'amazon_seller_central',
        'Salesforce': 'salesforce'
    }
    
    selected_platforms = []
    
    st.markdown("**Select your marketing platforms:**")
    cols = st.columns(3)
    
    for i, (platform_name, platform_id) in enumerate(platforms.items()):
        with cols[i % 3]:
            if st.checkbox(platform_name, key=f"platform_{platform_id}"):
                selected_platforms.append(platform_id)
    
    if selected_platforms:
        st.success(f"✅ Selected {len(selected_platforms)} platforms for automatic campaign detection")
    else:
        st.info("💡 You can connect platforms later in the Platform Connections page")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.wizard_step = 4
            st.rerun()
    with col3:
        if st.button("🎉 Complete Setup"):
            st.session_state.wizard_data['connected_platforms'] = selected_platforms
            complete_wizard_setup()
            st.rerun()

def complete_wizard_setup():
    """Complete the wizard setup and generate calendar configuration"""
    try:
        # Create wizard object
        wizard = CalendarSetupWizard(
            business_name=st.session_state.wizard_data['business_name'],
            industry=IndustryType(st.session_state.wizard_data['industry']),
            fiscal_year_start=FiscalYearStart(st.session_state.wizard_data['fiscal_year_start']),
            key_sales_periods=st.session_state.wizard_data.get('key_sales_periods', []),
            seasonal_patterns=st.session_state.wizard_data.get('seasonal_patterns', []),
            track_competitors=st.session_state.wizard_data.get('track_competitors', False),
            main_competitors=st.session_state.wizard_data.get('main_competitors', []),
            connected_platforms=st.session_state.wizard_data.get('connected_platforms', [])
        )
        
        # Generate configuration
        config = st.session_state.calendar_builder.create_from_wizard(wizard)
        st.session_state.calendar_config = config
        st.session_state.wizard_step = 6  # Completion step
        
        st.success("🎉 Calendar setup completed successfully!")
        
    except Exception as e:
        st.error(f"Error completing setup: {str(e)}")

def render_calendar_overview():
    """Render calendar overview after setup"""
    if not st.session_state.calendar_config:
        st.warning("Please complete the setup wizard first.")
        return
    
    config = st.session_state.calendar_config
    
    st.header("📅 Your Marketing Calendar")
    st.markdown(f"**Organization**: {config.organization_name}")
    st.markdown(f"**Industry**: {config.industry.value.replace('_', ' ').title()}")
    st.markdown(f"**Fiscal Year**: Starts in {config.fiscal_year_start.value.title()}")
    
    # Calendar summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Custom Events", len(config.custom_events))
    
    with col2:
        st.metric("Connected Platforms", len(config.campaign_platforms))
    
    with col3:
        competitor_count = len(config.competitors) if config.enable_competitor_tracking else 0
        st.metric("Tracked Competitors", competitor_count)
    
    with col4:
        st.metric("Auto-Detection", "Enabled" if config.auto_detect_campaigns else "Disabled")
    
    # Events timeline
    if config.custom_events:
        render_events_timeline(config.custom_events)
    
    # Calendar visualization
    render_calendar_visualization()

def render_events_timeline(events: List[BusinessEvent]):
    """Render events timeline"""
    st.subheader("📊 Business Events Timeline")
    
    # Create timeline data
    timeline_data = []
    for event in events:
        timeline_data.append({
            'Event': event.name,
            'Start': event.start_date,
            'End': event.end_date or event.start_date,
            'Type': event.event_type.value.replace('_', ' ').title(),
            'Impact': event.impact_level.value.title(),
            'Duration': (event.end_date - event.start_date).days + 1 if event.end_date else 1
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="End",
            y="Event",
            color="Impact",
            title="Business Events Timeline",
            color_discrete_map={
                'High': '#dc3545',
                'Medium': '#ffc107',
                'Low': '#28a745'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_calendar_visualization():
    """Render interactive calendar visualization"""
    st.subheader("📅 Interactive Calendar View")
    
    # Month selector
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Year", [2024, 2025, 2026], index=0)
    with col2:
        selected_month = st.selectbox("Month", range(1, 13), 
                                    format_func=lambda x: calendar.month_name[x])
    
    # Generate calendar for selected month
    cal = calendar.monthcalendar(selected_year, selected_month)
    
    # Create calendar grid
    st.markdown(f"### {calendar.month_name[selected_month]} {selected_year}")
    
    # Days of week header
    days_header = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    cols = st.columns(7)
    for i, day in enumerate(days_header):
        with cols[i]:
            st.markdown(f"**{day}**")
    
    # Calendar days
    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.write("")
                else:
                    # Check for events on this day
                    day_date = date(selected_year, selected_month, day)
                    events_on_day = []
                    
                    if st.session_state.calendar_config:
                        for event in st.session_state.calendar_config.custom_events:
                            if event.start_date <= day_date <= (event.end_date or event.start_date):
                                events_on_day.append(event)
                    
                    if events_on_day:
                        # Show day with events
                        event_names = [e.name[:10] + "..." if len(e.name) > 10 else e.name for e in events_on_day]
                        st.markdown(f"**{day}**")
                        for event_name in event_names:
                            st.markdown(f"<small>{event_name}</small>", unsafe_allow_html=True)
                    else:
                        st.write(str(day))

def render_natural_language_interface():
    """Render natural language query interface"""
    st.header("💬 Ask About Your Calendar")
    st.markdown("Ask questions about your marketing calendar in plain English!")
    
    # Query input
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., 'Show me the impact of Black Friday campaigns' or 'When do competitor launches affect us?'"
    )
    
    if query and st.button("Get Insights"):
        # Process query
        processor = NaturalLanguageProcessor()
        query_obj = ExecutiveCalendarQuery(query=query)
        
        # Generate sample calendar data for demo
        if st.session_state.calendar_config:
            calendar_data = st.session_state.calendar_builder.generate_calendar_dimension(
                st.session_state.calendar_config,
                date(2024, 1, 1),
                date(2024, 12, 31)
            )
            
            insights = processor.process_query(query_obj, calendar_data)
            
            if insights:
                st.subheader("📊 Insights")
                for insight in insights:
                    with st.expander(f"💡 {insight.title}"):
                        st.markdown(f"**Summary**: {insight.summary}")
                        st.markdown(f"**Impact**: {insight.impact}")
                        st.markdown(f"**Confidence**: {insight.confidence:.1%}")
                        
                        if insight.recommendation:
                            st.success(f"**Recommendation**: {insight.recommendation}")
                        
                        if insight.supporting_data:
                            st.markdown("**Supporting Data**:")
                            for key, value in insight.supporting_data.items():
                                if isinstance(value, float):
                                    if value < 1:
                                        st.write(f"- {key.replace('_', ' ').title()}: {value:.1%}")
                                    else:
                                        st.write(f"- {key.replace('_', ' ').title()}: {value:.1f}")
                                else:
                                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                st.info("I couldn't find specific insights for that query. Try asking about Black Friday, holiday campaigns, or competitor impact.")

def render_event_manager():
    """Render event management interface"""
    st.header("📝 Manage Business Events")
    
    # Add new event
    with st.expander("➕ Add New Event"):
        col1, col2 = st.columns(2)
        
        with col1:
            event_name = st.text_input("Event Name", placeholder="e.g., Spring Sale Campaign")
            event_type = st.selectbox("Event Type", [e.value for e in EventType],
                                    format_func=lambda x: x.replace('_', ' ').title())
            impact_level = st.selectbox("Impact Level", [e.value for e in EventImpact],
                                      format_func=lambda x: x.title())
        
        with col2:
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date (optional)", value=None)
            description = st.text_area("Description", placeholder="Event description...")
        
        if st.button("Add Event"):
            if event_name and start_date:
                new_event = BusinessEvent(
                    name=event_name,
                    start_date=start_date,
                    end_date=end_date,
                    event_type=EventType(event_type),
                    impact_level=EventImpact(impact_level),
                    description=description
                )
                st.session_state.custom_events.append(new_event)
                st.success(f"Added event: {event_name}")
                st.rerun()
    
    # Display existing events
    if st.session_state.custom_events:
        st.subheader("Current Events")
        
        events_data = []
        for i, event in enumerate(st.session_state.custom_events):
            events_data.append({
                'Name': event.name,
                'Type': event.event_type.value.replace('_', ' ').title(),
                'Start Date': event.start_date.strftime('%Y-%m-%d'),
                'End Date': event.end_date.strftime('%Y-%m-%d') if event.end_date else 'Single Day',
                'Impact': event.impact_level.value.title(),
                'Description': event.description or 'No description'
            })
        
        df = pd.DataFrame(events_data)
        st.dataframe(df, use_container_width=True)

def main():
    """Main application function"""
    # Demo mode - skip authentication
    st.info("🚀 Demo Mode: Executive Calendar Builder")
    
    initialize_session_state()
    
    st.title("📅 Executive Calendar Builder")
    st.markdown("Build your marketing calendar in minutes, not hours!")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.calendar_config is None:
        page = st.sidebar.radio("Setup", ["Setup Wizard"])
    else:
        page = st.sidebar.radio(
            "Calendar Management",
            ["Calendar Overview", "Natural Language Queries", "Manage Events", "Setup Wizard"]
        )
    
    # Reset wizard button
    if st.sidebar.button("🔄 Start Over"):
        st.session_state.wizard_step = 1
        st.session_state.wizard_data = {}
        st.session_state.calendar_config = None
        st.session_state.custom_events = []
        st.rerun()
    
    # Main content
    try:
        if page == "Setup Wizard":
            if st.session_state.wizard_step <= 5:
                render_setup_wizard()
            else:
                render_calendar_overview()
        elif page == "Calendar Overview":
            render_calendar_overview()
        elif page == "Natural Language Queries":
            render_natural_language_interface()
        elif page == "Manage Events":
            render_event_manager()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()