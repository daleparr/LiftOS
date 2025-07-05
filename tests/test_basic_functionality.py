"""
Basic functionality tests that don't require running services
"""
import pytest
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_module_configurations():
    """Test that all module configurations are valid"""
    modules_dir = project_root / "modules"
    
    for module_dir in modules_dir.iterdir():
        if module_dir.is_dir() and not module_dir.name.startswith('.'):
            module_json_path = module_dir / "module.json"
            
            # Check that module.json exists
            assert module_json_path.exists(), f"module.json missing in {module_dir.name}"
            
            # Check that module.json is valid JSON
            with open(module_json_path, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            assert "name" in config, f"Missing 'name' in {module_dir.name}/module.json"
            assert "version" in config, f"Missing 'version' in {module_dir.name}/module.json"
            
            # Check description (either top-level or in metadata)
            has_description = (
                "description" in config or 
                ("metadata" in config and "description" in config["metadata"])
            )
            assert has_description, f"Missing 'description' in {module_dir.name}/module.json"

def test_service_configurations():
    """Test that all service configurations are valid"""
    services_dir = project_root / "services"
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            # Check required files exist
            required_files = ["app.py", "requirements.txt", "Dockerfile"]
            for file_name in required_files:
                file_path = service_dir / file_name
                assert file_path.exists(), f"Missing {file_name} in {service_dir.name}"

def test_environment_configuration():
    """Test environment configuration"""
    env_example_path = project_root / ".env.example"
    assert env_example_path.exists(), "Missing .env.example file"
    
    # Check that .env.example has required variables
    with open(env_example_path, 'r') as f:
        env_content = f.read()
    
    required_vars = [
        "DATABASE_URL",
        "JWT_SECRET",
        "REDIS_URL"
    ]
    
    for var in required_vars:
        assert var in env_content, f"Missing {var} in .env.example"

def test_docker_configurations():
    """Test Docker configurations"""
    docker_compose_path = project_root / "docker-compose.yml"
    docker_compose_prod_path = project_root / "docker-compose.prod.yml"
    
    assert docker_compose_path.exists(), "Missing docker-compose.yml"
    assert docker_compose_prod_path.exists(), "Missing docker-compose.prod.yml"

def test_kubernetes_configurations():
    """Test Kubernetes configurations"""
    k8s_dir = project_root / "k8s"
    assert k8s_dir.exists(), "Missing k8s directory"
    
    required_k8s_files = [
        "namespace.yaml",
        "configmap.yaml",
        "secrets.yaml",
        "postgres.yaml",
        "redis.yaml",
        "ingress.yaml"
    ]
    
    for file_name in required_k8s_files:
        file_path = k8s_dir / file_name
        assert file_path.exists(), f"Missing {file_name} in k8s/"

def test_documentation_exists():
    """Test that documentation exists"""
    docs_dir = project_root / "docs"
    assert docs_dir.exists(), "Missing docs directory"
    
    required_docs = [
        "API.md",
        "DEPLOYMENT.md",
        "DEVELOPER_GUIDE.md"
    ]
    
    for doc_name in required_docs:
        doc_path = docs_dir / doc_name
        assert doc_path.exists(), f"Missing {doc_name} in docs/"
        
        # Check that docs have content
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert len(content) > 100, f"{doc_name} appears to be empty or too short"

def test_ui_shell_configuration():
    """Test UI shell configuration"""
    ui_dir = project_root / "ui-shell"
    assert ui_dir.exists(), "Missing ui-shell directory"
    
    required_files = [
        "package.json",
        "next.config.js",
        "tailwind.config.js",
        "tsconfig.json"
    ]
    
    for file_name in required_files:
        file_path = ui_dir / file_name
        assert file_path.exists(), f"Missing {file_name} in ui-shell/"

def test_shared_library_structure():
    """Test shared library structure"""
    shared_dir = project_root / "shared"
    assert shared_dir.exists(), "Missing shared directory"
    
    # Check for Python package structure
    init_file = shared_dir / "__init__.py"
    assert init_file.exists(), "Missing __init__.py in shared/"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])