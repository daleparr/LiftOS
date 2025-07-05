#!/usr/bin/env python3
"""
Quick test to verify the import fix works
"""
import sys
import os
sys.path.append('.')

try:
    from shared.kse_sdk.client import kse_client
    print("[OK] Successfully imported kse_client")
    
    # Check if it's using the real pinecone client
    client_type = type(kse_client.pinecone_client).__name__
    print(f"[OK] Using client type: {client_type}")
    
    # Check if hybrid_search method has correct signature
    import inspect
    sig = inspect.signature(kse_client.pinecone_client.hybrid_search)
    params = list(sig.parameters.keys())
    print(f"[OK] hybrid_search parameters: {params}")
    
    if 'search_type' in params:
        print("[SUCCESS] hybrid_search method has search_type parameter!")
    else:
        print("[FAIL] hybrid_search method missing search_type parameter")
        
except Exception as e:
    print(f"[FAIL] Import failed: {e}")