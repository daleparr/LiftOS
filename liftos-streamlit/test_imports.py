#!/usr/bin/env python3
"""Test script to verify imports work correctly"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from auth.authenticator import authenticate_user
    print("SUCCESS: Successfully imported authenticate_user from auth.authenticator")
except ImportError as e:
    print(f"ERROR: Failed to import authenticate_user: {e}")

try:
    from auth.session_manager import initialize_session
    print("SUCCESS: Successfully imported initialize_session from auth.session_manager")
except ImportError as e:
    print(f"ERROR: Failed to import initialize_session: {e}")

try:
    from utils.api_client import APIClient
    print("SUCCESS: Successfully imported APIClient from utils.api_client")
except ImportError as e:
    print(f"ERROR: Failed to import APIClient: {e}")

print("Import test completed.")