@echo off
echo ========================================
echo LiftOS Causal AI Service Setup
echo ========================================
echo.

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

:: Check if external/causal directory exists
if not exist "external\causal" (
    echo ERROR: external\causal directory not found.
    echo Please clone the causal AI repository to external\causal first:
    echo git clone https://github.com/daleparr/lift-causal-ai external/causal
    pause
    exit /b 1
)

echo Step 1: Building Causal AI service Docker image...
docker build -f Dockerfile.causal-service -t liftos-causal-service .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Causal AI service image
    pause
    exit /b 1
)

echo Step 2: Building Causal AI module Docker image...
docker build -f modules\causal\Dockerfile -t liftos-causal-module .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Causal AI module image
    pause
    exit /b 1
)

echo Step 3: Starting Causal AI services...
docker-compose -f docker-compose.production.yml up -d causal-service causal
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Causal AI services
    pause
    exit /b 1
)

echo Step 4: Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo Step 5: Testing Causal AI service health...
curl -f http://localhost:3003/health >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Causal AI service health check failed
    echo Service may still be starting up...
)

echo Step 6: Testing Causal AI module health...
curl -f http://localhost:8008/health >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Causal AI module health check failed
    echo Module may still be starting up...
)

echo Step 7: Registering Causal AI module with LiftOS registry...
python scripts\register_causal_module.py
if %errorlevel% neq 0 (
    echo WARNING: Failed to register Causal AI module
    echo You may need to register it manually later
)

echo.
echo ========================================
echo Causal AI Service Setup Complete!
echo ========================================
echo.
echo Services running:
echo - Causal AI Service: http://localhost:3003
echo - Causal AI Module: http://localhost:8008
echo.
echo To view logs:
echo   docker-compose logs causal-service
echo   docker-compose logs causal
echo.
echo To stop services:
echo   docker-compose down causal-service causal
echo.
pause