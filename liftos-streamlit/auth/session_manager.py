import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

def initialize_session():
    """Initialize session state variables"""
    
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None
    
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    
    # Application state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}
    
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()
    
    # Data state
    if 'meta_data' not in st.session_state:
        st.session_state.meta_data = None
    
    if 'google_data' not in st.session_state:
        st.session_state.google_data = None
    
    if 'klaviyo_data' not in st.session_state:
        st.session_state.klaviyo_data = None
    
    if 'attribution_results' not in st.session_state:
        st.session_state.attribution_results = None

def update_last_activity():
    """Update last activity timestamp"""
    st.session_state.last_activity = datetime.now()

def is_session_valid(timeout_minutes: int = 60) -> bool:
    """Check if session is still valid"""
    if not st.session_state.authenticated:
        return False
    
    if st.session_state.last_activity is None:
        return False
    
    timeout_delta = timedelta(minutes=timeout_minutes)
    return datetime.now() - st.session_state.last_activity < timeout_delta

def login_user(username: str, user_id: str, auth_token: Optional[str] = None):
    """Log in user and set session state"""
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.user_id = user_id
    st.session_state.auth_token = auth_token
    st.session_state.login_time = datetime.now()
    update_last_activity()

def logout_user():
    """Log out user and clear session state"""
    # Clear authentication state
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.auth_token = None
    st.session_state.login_time = None
    
    # Clear data state
    st.session_state.meta_data = None
    st.session_state.google_data = None
    st.session_state.klaviyo_data = None
    st.session_state.attribution_results = None
    
    # Clear cache
    st.session_state.api_cache = {}

def get_user_context() -> dict:
    """Get current user context"""
    return {
        'user_id': st.session_state.get('user_id'),
        'username': st.session_state.get('username'),
        'authenticated': st.session_state.get('authenticated', False),
        'login_time': st.session_state.get('login_time'),
        'last_activity': st.session_state.get('last_activity')
    }

def cache_api_response(key: str, data: dict, ttl_minutes: int = 5):
    """Cache API response with TTL"""
    cache_entry = {
        'data': data,
        'timestamp': datetime.now(),
        'ttl_minutes': ttl_minutes
    }
    st.session_state.api_cache[key] = cache_entry

def get_cached_response(key: str) -> Optional[dict]:
    """Get cached API response if still valid"""
    if key not in st.session_state.api_cache:
        return None
    
    cache_entry = st.session_state.api_cache[key]
    cache_time = cache_entry['timestamp']
    ttl_delta = timedelta(minutes=cache_entry['ttl_minutes'])
    
    if datetime.now() - cache_time > ttl_delta:
        # Cache expired, remove it
        del st.session_state.api_cache[key]
        return None
    
    return cache_entry['data']

def clear_cache():
    """Clear all cached data"""
    st.session_state.api_cache = {}