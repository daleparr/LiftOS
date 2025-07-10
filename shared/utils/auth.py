"""
Authentication utilities for Streamlit pages
"""

import streamlit as st
from typing import Dict, Any, Optional
from functools import wraps

def get_user_context() -> Dict[str, Any]:
    """Get user context from session state or create demo context"""
    if 'user_context' not in st.session_state:
        # Create demo user context for development
        st.session_state.user_context = {
            'user_id': 'demo_user_123',
            'org_id': 'demo_org_456',
            'username': 'Demo User',
            'email': 'demo@liftos.com',
            'roles': ['admin', 'user'],
            'is_authenticated': True
        }
    
    return st.session_state.user_context

def require_auth(func):
    """Decorator to require authentication for Streamlit pages"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        user_context = get_user_context()
        
        if not user_context.get('is_authenticated', False):
            st.error("Authentication required. Please log in.")
            st.stop()
        
        return func(*args, **kwargs)
    
    return wrapper

def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests"""
    user_context = get_user_context()
    
    return {
        'X-User-ID': user_context.get('user_id', ''),
        'X-Org-ID': user_context.get('org_id', ''),
        'X-User-Roles': ','.join(user_context.get('roles', [])),
        'Content-Type': 'application/json'
    }

def set_user_context(user_id: str, org_id: str, username: str, email: str, roles: list):
    """Set user context in session state"""
    st.session_state.user_context = {
        'user_id': user_id,
        'org_id': org_id,
        'username': username,
        'email': email,
        'roles': roles,
        'is_authenticated': True
    }

def logout():
    """Clear user context and log out"""
    if 'user_context' in st.session_state:
        del st.session_state.user_context
    
    st.success("Logged out successfully")
    st.rerun()

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    user_context = get_user_context()
    return user_context.get('is_authenticated', False)

def has_role(role: str) -> bool:
    """Check if user has specific role"""
    user_context = get_user_context()
    return role in user_context.get('roles', [])

def get_user_id() -> str:
    """Get current user ID"""
    user_context = get_user_context()
    return user_context.get('user_id', '')

def get_org_id() -> str:
    """Get current organization ID"""
    user_context = get_user_context()
    return user_context.get('org_id', '')