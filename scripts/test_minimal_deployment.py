#!/usr/bin/env python3
"""
LiftOS Minimal Infrastructure Test
Tests basic PostgreSQL and Redis connectivity
"""

import subprocess
import time
import sys
import psycopg2
import redis

def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def test_postgres_connection():
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            database="liftos",
            user="liftos"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        print(f"[OK] PostgreSQL connected successfully: {version[0][:50]}...")
        return True
    except Exception as e:
        print(f"[FAIL] PostgreSQL connection failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        r.delete('test_key')
        print(f"[OK] Redis connected successfully: ping and set/get operations work")
        return True
    except Exception as e:
        print(f"[FAIL] Redis connection failed: {e}")
        return False

def main():
    print("LiftOS Minimal Infrastructure Test")
    print("=" * 50)
    
    # Check if containers are already running
    print("\nChecking if containers are already running...")
    success, stdout, stderr = run_command("docker-compose -f docker-compose.minimal.yml ps", check=False)
    
    if success and "Up" in stdout:
        print("Containers are already running, skipping restart...")
    else:
        # Stop any existing containers
        print("Stopping existing containers...")
        run_command("docker-compose -f docker-compose.minimal.yml down", check=False)
        
        # Start minimal services
        print("Starting minimal services (PostgreSQL + Redis)...")
        success, stdout, stderr = run_command("docker-compose -f docker-compose.minimal.yml up -d")
        
        if not success:
            print(f"ERROR: Failed to start services: {stderr}")
            return False
        
        print("Services started, waiting for health checks...")
    
    # Wait for services to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        print(f"Checking services... (attempt {attempt + 1}/{max_attempts})")
        
        # Check if containers are running
        success, stdout, stderr = run_command("docker-compose -f docker-compose.minimal.yml ps")
        if success and "Up" in stdout:
            print("Containers are running, testing connections...")
            time.sleep(2)  # Give services a moment to fully initialize
            break
        
        time.sleep(2)
    else:
        print("ERROR: Services failed to start within timeout")
        return False
    
    # Test connections
    print("\nTesting database connections...")
    postgres_ok = test_postgres_connection()
    redis_ok = test_redis_connection()
    
    # Show container status
    print("\nContainer Status:")
    run_command("docker-compose -f docker-compose.minimal.yml ps")
    
    if postgres_ok and redis_ok:
        print("\n[SUCCESS] Minimal infrastructure is working correctly!")
        print("PostgreSQL and Redis are both accessible and functional.")
        return True
    else:
        print("\n[FAILURE] Some services are not working correctly")
        print("Check the logs above for details.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)