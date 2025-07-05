import streamlit as st
import requests
from typing import Optional, Dict
from config.settings import get_auth_config
from auth.session_manager import login_user, logout_user, is_session_valid, update_last_activity

class StreamlitAuthenticator:
    """Streamlit authentication handler"""
    
    def __init__(self):
        self.auth_config = get_auth_config()
        self.auth_service_url = self.auth_config['auth_service_url']
        self.require_auth = self.auth_config['require_auth']
        self.demo_mode = self.auth_config['demo_mode']
    
    def authenticate_user(self) -> bool:
        """Authenticate user and manage session"""
        
        # Check if already authenticated and session is valid
        if st.session_state.get('authenticated', False):
            if is_session_valid():
                update_last_activity()
                return True
            else:
                # Session expired
                logout_user()
                st.warning("Session expired. Please log in again.")
        
        # Demo mode - skip authentication
        if self.demo_mode and not self.require_auth:
            return self._demo_login()
        
        # Show login form
        return self._show_login_form()
    
    def _demo_login(self) -> bool:
        """Demo mode login"""
        if not st.session_state.get('authenticated', False):
            login_user(
                username="demo_user",
                user_id="demo_user_123",
                auth_token="demo_token"
            )
            st.success("Demo mode - automatically logged in!")
        return True
    
    def _show_login_form(self) -> bool:
        """Show login form in sidebar"""
        with st.sidebar:
            st.subheader("🔐 Login to LiftOS")
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_button = st.form_submit_button("Login", use_container_width=True)
                with col2:
                    demo_button = st.form_submit_button("Demo Mode", use_container_width=True)
                
                if login_button and username and password:
                    if self._validate_credentials(username, password):
                        user_id = self._get_user_id(username)
                        auth_token = self._get_auth_token(username, password)
                        login_user(username, user_id, auth_token)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                
                if demo_button:
                    login_user(
                        username="demo_user",
                        user_id="demo_user_123",
                        auth_token="demo_token"
                    )
                    st.success("Demo mode activated!")
                    st.rerun()
            
            # Help text
            st.markdown("---")
            st.markdown("**Demo Mode**: Try LiftOS without authentication")
            st.markdown("**Need Help?** Contact support@liftos.ai")
        
        return False
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials with auth service"""
        try:
            response = requests.post(
                f"{self.auth_service_url}/api/v1/auth/login",
                json={"username": username, "password": password},
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            st.error(f"Authentication service unavailable: {e}")
            # Allow demo login if auth service is down
            if username == "admin" and password == "admin":
                return True
            return False
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return False
    
    def _get_user_id(self, username: str) -> str:
        """Get user ID from username"""
        try:
            response = requests.get(
                f"{self.auth_service_url}/api/v1/users/{username}",
                timeout=10
            )
            if response.status_code == 200:
                user_data = response.json()
                return user_data.get('user_id', f"user_{username}")
        except:
            pass
        
        # Fallback
        return f"user_{username}"
    
    def _get_auth_token(self, username: str, password: str) -> Optional[str]:
        """Get authentication token"""
        try:
            response = requests.post(
                f"{self.auth_service_url}/api/v1/auth/token",
                json={"username": username, "password": password},
                timeout=10
            )
            if response.status_code == 200:
                token_data = response.json()
                return token_data.get('access_token')
        except:
            pass
        
        # Return demo token for fallback
        return f"token_{username}"

def authenticate_user() -> bool:
    """Main authentication function"""
    authenticator = StreamlitAuthenticator()
    return authenticator.authenticate_user()

def show_logout_button():
    """Show logout button in sidebar"""
    if st.session_state.get('authenticated', False):
        with st.sidebar:
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.success("Logged out successfully!")
                st.rerun()