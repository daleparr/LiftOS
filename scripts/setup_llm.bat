@echo off
echo ========================================
echo LiftOS LLM Service Setup
echo ========================================
echo.

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

:: Check if external/llm directory exists
if not exist "external\llm" (
    echo ERROR: external\llm directory not found.
    echo Please clone the LLM repository to external\llm first:
    echo git clone https://github.com/daleparr/Lift-os-LLM external/llm
    pause
    exit /b 1
)

echo Step 1: Building LLM service Docker image...
docker build -f Dockerfile.llm-service -t liftos-llm-service .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build LLM service image
    pause
    exit /b 1
)

echo Step 2: Building LLM module Docker image...
docker build -f modules\llm\Dockerfile -t liftos-llm-module .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build LLM module image
    pause
    exit /b 1
)

echo Step 3: Starting LLM services...
docker-compose -f docker-compose.production.yml up -d llm-service llm
if %errorlevel% neq 0 (
    echo ERROR: Failed to start LLM services
    pause
    exit /b 1
)

echo Step 4: Waiting for services to be ready...
timeout /t 15 /nobreak >nul

echo Step 5: Testing LLM service health...
curl -f http://localhost:3004/health >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: LLM service health check failed
    echo Service may still be starting up...
)

echo Step 6: Testing LLM module health...
curl -f http://localhost:8009/health >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: LLM module health check failed
    echo Module may still be starting up...
)

echo Step 7: Registering LLM module with LiftOS registry...
python scripts\register_llm_module.py
if %errorlevel% neq 0 (
    echo WARNING: Failed to register LLM module
    echo You may need to register it manually later
)

echo.
echo ========================================
echo LLM Service Setup Complete!
echo ========================================
echo.
echo Services running:
echo - LLM Service: http://localhost:3004
echo - LLM Module: http://localhost:8009
echo.
echo Available capabilities:
echo - Model evaluation and leaderboard
echo - Content generation with prompt templates
echo - Multi-provider LLM integration (OpenAI, Cohere, HuggingFace)
echo - Evaluation metrics (BLEU, ROUGE, BERTScore, RLHF)
echo - Multilingual support and context-length testing
echo.
echo To view logs:
echo   docker-compose logs llm-service
echo   docker-compose logs llm
echo.
echo To stop services:
echo   docker-compose down llm-service llm
echo.
echo Next steps:
echo 1. Configure API keys in .env.production:
echo    - OPENAI_API_KEY=your_openai_key
echo    - COHERE_API_KEY=your_cohere_key
echo    - HUGGINGFACE_API_KEY=your_hf_key
echo 2. Run integration tests: python scripts\test_llm_integration.py
echo 3. Access LLM module via API Gateway: http://localhost:8000/modules/llm/
echo.
pause