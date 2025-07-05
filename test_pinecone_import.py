#!/usr/bin/env python3
"""
Test script to check pinecone import options
"""

print("Testing pinecone import options...")

try:
    from pinecone import Pinecone, ServerlessSpec
    print("✓ Successfully imported: from pinecone import Pinecone, ServerlessSpec")
except ImportError as e:
    print(f"✗ Failed to import from pinecone: {e}")

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    print("✓ Successfully imported: from pinecone.grpc import PineconeGRPC as Pinecone")
except ImportError as e:
    print(f"✗ Failed to import from pinecone.grpc: {e}")

try:
    import pinecone
    print(f"✓ Successfully imported pinecone module: {pinecone.__version__}")
except ImportError as e:
    print(f"✗ Failed to import pinecone module: {e}")

print("Test complete.")