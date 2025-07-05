import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print(f"Project root: {project_root}")

# Test the path calculation from shared/database/connection.py
shared_db_file = os.path.join(os.path.dirname(__file__), "shared", "database", "connection.py")
print(f"Shared database file: {shared_db_file}")

# Calculate project root from shared/database/connection.py perspective
connection_file = os.path.join(project_root, "shared", "database", "connection.py")
# Go up 3 levels: connection.py -> database -> shared -> project_root
project_root_from_connection = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(connection_file))))
db_path = os.path.join(project_root_from_connection, "data", "lift_os_dev.db")

print(f"Connection file: {connection_file}")
print(f"Project root from connection: {project_root_from_connection}")
print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}")

# Test the actual database URL
database_url = f"sqlite+aiosqlite:///{db_path}"
print(f"Database URL: {database_url}")