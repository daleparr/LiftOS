"""
Test service syntax and basic imports
"""
import pytest
import ast
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_service_python_syntax():
    """Test that all service Python files have valid syntax"""
    services_dir = project_root / "services"
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            app_py_path = service_dir / "app.py"
            
            if app_py_path.exists():
                # Test syntax by parsing the AST
                with open(app_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    ast.parse(content)
                    print(f"OK {service_dir.name}/app.py has valid syntax")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {service_dir.name}/app.py: {e}")

def test_module_python_syntax():
    """Test that all module Python files have valid syntax"""
    modules_dir = project_root / "modules"
    
    for module_dir in modules_dir.iterdir():
        if module_dir.is_dir() and not module_dir.name.startswith('.'):
            app_py_path = module_dir / "app.py"
            
            if app_py_path.exists():
                # Test syntax by parsing the AST
                with open(app_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    ast.parse(content)
                    print(f"OK {module_dir.name}/app.py has valid syntax")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {module_dir.name}/app.py: {e}")

def test_shared_library_imports():
    """Test that shared library can be imported"""
    try:
        # Test basic imports from shared library - just check if modules exist
        import shared.auth
        import shared.kse_sdk
        import shared.models
        import shared.utils
        print("OK Shared library imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import from shared library: {e}")

def test_fastapi_app_structure():
    """Test that FastAPI apps are properly structured"""
    services_dir = project_root / "services"
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            app_py_path = service_dir / "app.py"
            
            if app_py_path.exists():
                with open(app_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for FastAPI app creation
                assert "FastAPI(" in content, f"No FastAPI app found in {service_dir.name}/app.py"
                
                # Check for health endpoint
                assert "/health" in content, f"No health endpoint found in {service_dir.name}/app.py"
                
                print(f"OK {service_dir.name}/app.py has proper FastAPI structure")

def test_requirements_files():
    """Test that requirements files are properly formatted"""
    services_dir = project_root / "services"
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            req_path = service_dir / "requirements.txt"
            
            if req_path.exists():
                with open(req_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Check that file is not empty
                assert content, f"Empty requirements.txt in {service_dir.name}"
                
                # Check for FastAPI dependency
                lines = content.split('\n')
                has_fastapi = any('fastapi' in line.lower() for line in lines)
                assert has_fastapi, f"No FastAPI dependency in {service_dir.name}/requirements.txt"
                
                print(f"OK {service_dir.name}/requirements.txt is properly formatted")

def test_dockerfile_structure():
    """Test that Dockerfiles have proper structure"""
    services_dir = project_root / "services"
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            dockerfile_path = service_dir / "Dockerfile"
            
            if dockerfile_path.exists():
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic Dockerfile structure
                assert "FROM python:" in content, f"No Python base image in {service_dir.name}/Dockerfile"
                assert "COPY requirements.txt" in content, f"No requirements copy in {service_dir.name}/Dockerfile"
                assert "RUN pip install" in content, f"No pip install in {service_dir.name}/Dockerfile"
                assert "CMD" in content or "ENTRYPOINT" in content, f"No CMD/ENTRYPOINT in {service_dir.name}/Dockerfile"
                
                print(f"OK {service_dir.name}/Dockerfile has proper structure")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])