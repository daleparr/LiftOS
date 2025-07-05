@echo off
setlocal enabledelayedexpansion

echo ========================================
echo LiftOS Surfacing Module Setup
echo ========================================
echo.

REM Check if Docker is running
echo [1/8] Checking Docker status...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo ✓ Docker is running

REM Check if docker-compose is available
echo.
echo [2/8] Checking docker-compose availability...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: docker-compose is not available. Please install Docker Compose.
    pause
    exit /b 1
)
echo ✓ docker-compose is available

REM Clone the surfacing repository if it doesn't exist
echo.
echo [3/8] Setting up surfacing repository...
if not exist "external\surfacing" (
    echo Cloning surfacing repository...
    if not exist "external" mkdir external
    cd external
    git clone https://github.com/daleparr/Lift-os-surfacing.git surfacing
    if errorlevel 1 (
        echo ERROR: Failed to clone surfacing repository
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo ✓ Surfacing repository cloned
) else (
    echo ✓ Surfacing repository already exists
)

REM Build the surfacing service Docker image
echo.
echo [4/8] Building surfacing service Docker image...
docker build -f Dockerfile.surfacing-service -t liftos/surfacing-service:latest .
if errorlevel 1 (
    echo ERROR: Failed to build surfacing service Docker image
    pause
    exit /b 1
)
echo ✓ Surfacing service Docker image built

REM Build the surfacing module Docker image
echo.
echo [5/8] Building surfacing module Docker image...
docker build -f modules/surfacing/Dockerfile -t liftos/surfacing:latest modules/surfacing
if errorlevel 1 (
    echo ERROR: Failed to build surfacing module Docker image
    pause
    exit /b 1
)
echo ✓ Surfacing module Docker image built

REM Start the services
echo.
echo [6/8] Starting LiftOS services with surfacing integration...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
if errorlevel 1 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)
echo ✓ Services started

REM Wait for services to be ready
echo.
echo [7/8] Waiting for services to be ready...
timeout /t 30 /nobreak >nul
echo ✓ Services should be ready

REM Register the surfacing module
echo.
echo [8/8] Registering surfacing module...
python scripts/register_surfacing_module.py
if errorlevel 1 (
    echo WARNING: Module registration failed. You may need to register manually.
) else (
    echo ✓ Surfacing module registered successfully
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo The surfacing module has been integrated into LiftOS.
echo.
echo Services running:
echo - LiftOS Core: http://localhost:8000
echo - Surfacing Service: http://localhost:3002
echo - Surfacing Module: http://localhost:8007
echo.
echo To test the integration:
echo   python scripts/test_surfacing_integration.py
echo.
echo To stop all services:
echo   docker-compose down
echo.
pause