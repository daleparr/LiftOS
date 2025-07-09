"""
KSE-SDK Integration Diagnostic Tool
Validates the integration between legacy client and universal architecture
"""

import logging
from typing import Dict, Any
from datetime import datetime

# Test imports to validate architecture
try:
    # Universal architecture imports
    from .core import KSEMemory, Entity, SearchQuery, SearchResult, DOMAIN_DIMENSIONS
    from .core.config import KSEConfig
    universal_architecture_available = True
    universal_import_error = None
except Exception as e:
    universal_architecture_available = False
    universal_import_error = str(e)

try:
    # Legacy client imports
    from .client import LiftKSEClient, kse_client
    from .models import MemorySearchResult, MemoryInsights
    legacy_client_available = True
    legacy_import_error = None
except Exception as e:
    legacy_client_available = False
    legacy_import_error = str(e)

try:
    # Legacy Pinecone client imports
    from .pinecone_client import PineconeKSEClient
    pinecone_client_available = True
    pinecone_import_error = None
except Exception as e:
    pinecone_client_available = False
    pinecone_import_error = str(e)


def run_integration_diagnostic() -> Dict[str, Any]:
    """
    Run comprehensive diagnostic of KSE-SDK integration status.
    
    Returns:
        Diagnostic report with findings and recommendations
    """
    diagnostic_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "universal_architecture": {
            "available": universal_architecture_available,
            "error": universal_import_error,
            "components": {}
        },
        "legacy_client": {
            "available": legacy_client_available,
            "error": legacy_import_error,
            "components": {}
        },
        "integration_status": "unknown",
        "critical_issues": [],
        "recommendations": []
    }
    
    # Test universal architecture components
    if universal_architecture_available:
        try:
            # Test KSEMemory instantiation
            config = KSEConfig(app_name="diagnostic_test")
            memory = KSEMemory(config)
            diagnostic_report["universal_architecture"]["components"]["KSEMemory"] = "✅ Available"
            
            # Test Entity creation
            entity = Entity(
                id="test_entity",
                content="Test content",
                domain="retail"
            )
            diagnostic_report["universal_architecture"]["components"]["Entity"] = "✅ Available"
            
            # Test domain dimensions
            retail_dimensions = DOMAIN_DIMENSIONS.get("retail", {})
            diagnostic_report["universal_architecture"]["components"]["DOMAIN_DIMENSIONS"] = f"✅ Available ({len(DOMAIN_DIMENSIONS)} domains)"
            
        except Exception as e:
            diagnostic_report["universal_architecture"]["components"]["instantiation_error"] = str(e)
    
    # Test legacy client components
    if legacy_client_available:
        try:
            # Test legacy client instantiation
            client = LiftKSEClient()
            diagnostic_report["legacy_client"]["components"]["LiftKSEClient"] = "✅ Available"
            
            # Check if it uses PineconeKSEClient
            if hasattr(client, 'pinecone_client'):
                diagnostic_report["legacy_client"]["components"]["uses_pinecone_client"] = "⚠️ Yes (Legacy)"
            
        except Exception as e:
            diagnostic_report["legacy_client"]["components"]["instantiation_error"] = str(e)
    
    # Analyze integration status
    if universal_architecture_available and legacy_client_available:
        # Check if legacy client uses universal architecture
        try:
            from .client import LiftKSEClient
            import inspect
            
            # Check imports in client module
            client_source = inspect.getsource(LiftKSEClient)
            
            if "KSEMemory" in client_source:
                diagnostic_report["integration_status"] = "✅ Integrated"
            elif "PineconeKSEClient" in client_source:
                diagnostic_report["integration_status"] = "❌ Legacy (Not Integrated)"
                diagnostic_report["critical_issues"].append(
                    "Legacy client still uses PineconeKSEClient instead of universal KSEMemory"
                )
            else:
                diagnostic_report["integration_status"] = "⚠️ Unknown"
                
        except Exception as e:
            diagnostic_report["integration_status"] = f"❌ Analysis Failed: {e}"
    
    # Generate recommendations
    if diagnostic_report["integration_status"] == "❌ Legacy (Not Integrated)":
        diagnostic_report["recommendations"].extend([
            "1. Rewrite shared/kse_sdk/client.py to use universal KSEMemory instead of PineconeKSEClient",
            "2. Update client to use universal Entity and SearchResult models",
            "3. Enable multi-backend support through universal adapters",
            "4. Migrate legacy MemorySearchResult to universal SearchResult",
            "5. Test all microservices with updated universal client"
        ])
    
    if not universal_architecture_available:
        diagnostic_report["critical_issues"].append(
            "Universal KSE-SDK architecture not available - core implementation missing"
        )
    
    if not legacy_client_available:
        diagnostic_report["critical_issues"].append(
            "Legacy client not available - services cannot access KSE-SDK"
        )
    
    return diagnostic_report


def log_diagnostic_report(report: Dict[str, Any]):
    """Log diagnostic report with appropriate levels."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== KSE-SDK Integration Diagnostic Report ===")
    logger.info(f"Timestamp: {report['timestamp']}")
    logger.info(f"Integration Status: {report['integration_status']}")
    
    if report["critical_issues"]:
        logger.error("Critical Issues Found:")
        for issue in report["critical_issues"]:
            logger.error(f"  - {issue}")
    
    if report["recommendations"]:
        logger.warning("Recommendations:")
        for rec in report["recommendations"]:
            logger.warning(f"  {rec}")
    
    logger.info("Universal Architecture:")
    for component, status in report["universal_architecture"]["components"].items():
        logger.info(f"  {component}: {status}")
    
    logger.info("Legacy Client:")
    for component, status in report["legacy_client"]["components"].items():
        logger.info(f"  {component}: {status}")


if __name__ == "__main__":
    # Run diagnostic when executed directly
    logging.basicConfig(level=logging.INFO)
    report = run_integration_diagnostic()
    log_diagnostic_report(report)
    
    # Print summary
    print("\n" + "="*60)
    print("KSE-SDK INTEGRATION DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Status: {report['integration_status']}")
    print(f"Critical Issues: {len(report['critical_issues'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
    print("="*60)