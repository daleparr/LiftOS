#!/usr/bin/env python3
"""
Direct test of hybrid search functionality with fixed API keys
"""
import requests
import json
import time

def test_hybrid_search():
    print("Testing Hybrid Search with Fixed Implementation")
    print("=" * 60)
    
    # Test memory storage first
    print("1. Testing memory storage...")
    store_data = {
        "content": "LiftOS hybrid search test memory",
        "metadata": {
            "source": "test_script",
            "type": "validation",
            "timestamp": time.time()
        },
        "memory_type": "knowledge"
    }
    
    try:
        response = requests.post(
            "http://localhost:8003/store",
            json=store_data,
            headers={
                "X-User-Id": "test_user",
                "X-Org-Id": "test_org",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            memory_id = response.json().get("memory_id")
            print(f"  [OK] Memory stored: {memory_id}")
        else:
            print(f"  [FAIL] Storage failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Storage error: {e}")
        return False
    
    # Test hybrid search
    print("2. Testing hybrid search...")
    search_data = {
        "query": "LiftOS hybrid search",
        "limit": 5,
        "search_type": "hybrid",
        "filters": {}
    }
    
    try:
        response = requests.post(
            "http://localhost:8003/search",
            json=search_data,
            headers={
                "X-User-Id": "test_user",
                "X-Org-Id": "test_org",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"  [OK] Search successful: {len(results.get('memories', []))} results")
            print(f"  [INFO] Search took: {results.get('search_time', 'unknown')}s")
            return True
        else:
            print(f"  [FAIL] Search failed: {response.status_code}")
            print(f"  [ERROR] Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Search error: {e}")
        return False

if __name__ == "__main__":
    success = test_hybrid_search()
    if success:
        print("\n[OK] Hybrid search test PASSED - fixes are working!")
    else:
        print("\n[FAIL] Hybrid search test FAILED - check logs")