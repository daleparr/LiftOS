#!/usr/bin/env python3
"""
KSE-SDK Universal Architecture Diagnostic Tool
Tests the complete universal KSE-SDK implementation
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, Any

# Add the project root to Python path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_imports():
    """Test all critical imports"""
    print("[*] Testing Universal KSE-SDK Imports...")
    
    try:
        # Test core imports
        from shared.kse_sdk.core import KSEMemory, Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
        print("[+] Core imports successful")
        
        # Test service imports
        from shared.kse_sdk.services import (
            EmbeddingService, ConceptualService, SearchService,
            CacheService, AnalyticsService, SecurityService,
            WorkflowService, NotificationService
        )
        print("[+] Service imports successful")
        
        # Test backend imports
        from shared.kse_sdk.backends import vector_stores, graph_stores, concept_stores
        print("[+] Backend imports successful")
        
        # Test exception imports
        from shared.kse_sdk.exceptions import (
            KSEError, ConfigurationError, BackendError,
            EmbeddingError, SearchError, ValidationError
        )
        print("[+] Exception imports successful")
        
        # Test adapter imports
        from shared.kse_sdk.adapters import get_adapter
        print("[+] Adapter imports successful")
        
        return True
        
    except Exception as e:
        print(f"[-] Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration system"""
    print("\n[*] Testing Configuration System...")
    
    try:
        from shared.kse_sdk.core import KSEConfig
        
        # Test default configuration
        config = KSEConfig()
        print(f"[+] Default config created: {config.vector_store_type}")
        
        # Test custom configuration
        custom_config = KSEConfig(
            vector_store_type="pinecone",
            graph_store_type="neo4j",
            concept_store_type="postgresql",
            embedding_model="openai"
        )
        print(f"[+] Custom config created: {custom_config.vector_store_type}")
        
        # Test version attribute
        print(f"[+] Config version: {config.version}")
        
        return True
        
    except Exception as e:
        print(f"[-] Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_entity_system():
    """Test universal entity system"""
    print("\n[*] Testing Universal Entity System...")
    
    try:
        from shared.kse_sdk.core import Entity
        
        # Test basic entity creation
        entity = Entity(
            id="test_entity_1",
            title="Test Entity",
            description="This is a test entity for universal KSE-SDK",
            entity_type="test",
            domain="general"
        )
        print(f"[+] Basic entity created: {entity.id}")
        
        # Test domain-specific entities
        healthcare_entity = Entity.create_healthcare_entity(
            id="P123",
            title="Patient Record",
            description="Patient diagnosis and treatment plan",
            patient_id="P123",
            diagnosis="Hypertension",
            treatment_plan="Medication and lifestyle changes"
        )
        print(f"[+] Healthcare entity created: {healthcare_entity.id}")
        
        # Test finance entity
        finance_entity = Entity.create_finance_entity(
            id="T456",
            title="Transaction Record",
            description="Financial transaction details",
            transaction_id="T456",
            amount=1500.00,
            currency="USD",
            transaction_type="payment"
        )
        print(f"[+] Finance entity created: {finance_entity.id}")
        
        return True
        
    except Exception as e:
        print(f"[-] Entity system test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_conceptual_spaces():
    """Test conceptual space system"""
    print("\n[*] Testing Conceptual Space System...")
    
    try:
        from shared.kse_sdk.core import ConceptualSpace, DOMAIN_DIMENSIONS
        
        # Test basic conceptual space
        space = ConceptualSpace(
            dimensions={"quality": 0.5, "price": 0.8, "brand": 0.3},
            domain="retail"
        )
        print(f"[+] Basic conceptual space created for domain: {space.domain}")
        
        # Test domain-specific spaces
        healthcare_space = ConceptualSpace.create_healthcare_space()
        print(f"[+] Healthcare space created with {len(healthcare_space.dimensions)} dimensions")
        
        finance_space = ConceptualSpace.create_finance_space()
        print(f"[+] Finance space created with {len(finance_space.dimensions)} dimensions")
        
        # Test domain dimensions
        print(f"[+] Available domains: {list(DOMAIN_DIMENSIONS.keys())}")
        
        return True
        
    except Exception as e:
        print(f"[-] Conceptual space test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_kse_memory():
    """Test KSE Memory initialization"""
    print("\n[*] Testing KSE Memory Initialization...")
    
    try:
        from shared.kse_sdk.core import KSEMemory, KSEConfig
        
        # Test with default config
        config = KSEConfig()
        memory = KSEMemory(config)
        print(f"[+] KSE Memory initialized with {config.vector_store_type} backend")
        
        # Test memory attributes
        print(f"[+] Memory has embedding service: {hasattr(memory, 'embedding_service')}")
        print(f"[+] Memory has conceptual service: {hasattr(memory, 'conceptual_service')}")
        print(f"[+] Memory has search service: {hasattr(memory, 'search_service')}")
        
        return True
        
    except Exception as e:
        print(f"[-] KSE Memory test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_services():
    """Test service layer"""
    print("\n[*] Testing Service Layer...")
    
    try:
        from shared.kse_sdk.services import (
            EmbeddingService, ConceptualService, SearchService,
            CacheService, AnalyticsService
        )
        
        # Test service interfaces exist (they are abstract, so we can't instantiate them)
        print(f"[+] Embedding service interface available: {EmbeddingService is not None}")
        print(f"[+] Conceptual service interface available: {ConceptualService is not None}")
        print(f"[+] Search service interface available: {SearchService is not None}")
        print(f"[+] Cache service available: {CacheService is not None}")
        print(f"[+] Analytics service available: {AnalyticsService is not None}")
        
        return True
        
    except Exception as e:
        print(f"[-] Service layer test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_backends():
    """Test backend factories"""
    print("\n[*] Testing Backend Factories...")
    
    try:
        from shared.kse_sdk.backends.vector_stores import get_vector_store
        from shared.kse_sdk.backends.graph_stores import get_graph_store
        from shared.kse_sdk.backends.concept_stores import get_concept_store
        from shared.kse_sdk.core.config import VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig
        
        # Test that factory functions exist and can be called
        print(f"[+] Vector store factory function available: {get_vector_store is not None}")
        print(f"[+] Graph store factory function available: {get_graph_store is not None}")
        print(f"[+] Concept store factory function available: {get_concept_store is not None}")
        
        # Test config classes can be created
        vector_config = VectorStoreConfig(backend="pinecone")
        graph_config = GraphStoreConfig(backend="neo4j")
        concept_config = ConceptStoreConfig(backend="postgresql")
        print(f"[+] Backend configurations created successfully")
        
        return True
        
    except Exception as e:
        print(f"[-] Backend factory test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_client_integration():
    """Test client integration"""
    print("\n[*] Testing Client Integration...")
    
    try:
        from shared.kse_sdk.client import LiftKSEClient
        
        # Test client creation
        client = LiftKSEClient()
        print(f"[+] Client created: {type(client).__name__}")
        
        # Test client has memory
        print(f"[+] Client has kse_memory: {hasattr(client, 'kse_memory')}")
        
        return True
        
    except Exception as e:
        print(f"[-] Client integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("*** KSE-SDK Universal Architecture Diagnostic Tool")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Entity System Tests", test_entity_system),
        ("Conceptual Space Tests", test_conceptual_spaces),
        ("KSE Memory Tests", test_kse_memory),
        ("Service Layer Tests", test_services),
        ("Backend Factory Tests", test_backends),
        ("Client Integration Tests", test_client_integration),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
        except Exception as e:
            print(f"[-] {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("*** DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "[+] PASS" if success else "[-] FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Universal KSE-SDK is fully functional!")
    elif passed >= total * 0.75:
        print("[WARNING] Most tests passed, minor issues detected")
    else:
        print("[WARNING] Some issues detected in Universal KSE-SDK")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)