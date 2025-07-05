from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import json
import asyncio
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lift Eval Module",
    description="AI Model Evaluation and Testing Framework",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EvaluationRequest(BaseModel):
    model_name: str
    test_dataset: str
    evaluation_metrics: List[str]
    parameters: Optional[Dict[str, Any]] = {}

class TestCase(BaseModel):
    id: str
    input_data: Dict[str, Any]
    expected_output: Any
    actual_output: Optional[Any] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}

class EvaluationResult(BaseModel):
    evaluation_id: str
    model_name: str
    test_dataset: str
    metrics: Dict[str, float]
    test_cases: List[TestCase]
    summary: Dict[str, Any]
    created_at: datetime

class BenchmarkRequest(BaseModel):
    models: List[str]
    benchmark_suite: str
    comparison_metrics: List[str]

# In-memory storage (replace with database in production)
evaluations_db = {}
benchmarks_db = {}

# Registry integration
async def register_with_registry():
    """Register this module with the registry service"""
    try:
        async with httpx.AsyncClient() as client:
            module_info = {
                "name": "lift-eval",
                "version": "1.0.0",
                "description": "AI Model Evaluation and Testing Framework",
                "status": "active",
                "endpoints": [
                    "/health",
                    "/evaluate",
                    "/benchmark",
                    "/results",
                    "/test-cases"
                ],
                "permissions": [
                    "eval:create",
                    "eval:read",
                    "eval:benchmark",
                    "models:test"
                ],
                "ui_components": [
                    {
                        "name": "EvaluationDashboard",
                        "type": "dashboard",
                        "path": "/eval/dashboard"
                    },
                    {
                        "name": "BenchmarkComparison",
                        "type": "widget",
                        "path": "/eval/benchmark"
                    }
                ]
            }
            
            response = await client.post(
                "http://registry:8003/api/v1/modules/register",
                json=module_info
            )
            
            if response.status_code == 200:
                logger.info("Successfully registered with registry")
            else:
                logger.error(f"Failed to register with registry: {response.text}")
                
    except Exception as e:
        logger.error(f"Error registering with registry: {e}")

# Memory integration
async def store_evaluation_memory(evaluation_id: str, result: EvaluationResult):
    """Store evaluation results in KSE Memory for future reference"""
    try:
        async with httpx.AsyncClient() as client:
            memory_data = {
                "content": f"Evaluation {evaluation_id} for model {result.model_name}",
                "metadata": {
                    "type": "evaluation_result",
                    "evaluation_id": evaluation_id,
                    "model_name": result.model_name,
                    "metrics": result.metrics,
                    "test_dataset": result.test_dataset,
                    "created_at": result.created_at.isoformat()
                },
                "tags": ["evaluation", "model_testing", result.model_name]
            }
            
            response = await client.post(
                "http://memory:8002/api/v1/memories",
                json=memory_data
            )
            
            if response.status_code == 200:
                logger.info(f"Stored evaluation {evaluation_id} in memory")
            else:
                logger.error(f"Failed to store evaluation in memory: {response.text}")
                
    except Exception as e:
        logger.error(f"Error storing evaluation in memory: {e}")

@app.on_event("startup")
async def startup_event():
    """Register with registry on startup"""
    await register_with_registry()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "lift-eval",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/evaluate", response_model=EvaluationResult)
async def create_evaluation(request: EvaluationRequest):
    """Create a new model evaluation"""
    try:
        evaluation_id = str(uuid.uuid4())
        
        # Simulate evaluation process
        test_cases = await generate_test_cases(request.test_dataset)
        evaluated_cases = await run_evaluation(request.model_name, test_cases, request.parameters)
        metrics = await calculate_metrics(evaluated_cases, request.evaluation_metrics)
        
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            model_name=request.model_name,
            test_dataset=request.test_dataset,
            metrics=metrics,
            test_cases=evaluated_cases,
            summary={
                "total_test_cases": len(evaluated_cases),
                "passed_cases": len([tc for tc in evaluated_cases if tc.score and tc.score > 0.7]),
                "average_score": sum(tc.score for tc in evaluated_cases if tc.score) / len(evaluated_cases),
                "evaluation_time": "45.2s"
            },
            created_at=datetime.utcnow()
        )
        
        # Store in database
        evaluations_db[evaluation_id] = result
        
        # Store in memory for future reference
        await store_evaluation_memory(evaluation_id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/benchmark")
async def create_benchmark(request: BenchmarkRequest):
    """Create a benchmark comparison between multiple models"""
    try:
        benchmark_id = str(uuid.uuid4())
        
        # Run evaluations for each model
        results = {}
        for model in request.models:
            eval_request = EvaluationRequest(
                model_name=model,
                test_dataset=request.benchmark_suite,
                evaluation_metrics=request.comparison_metrics
            )
            result = await create_evaluation(eval_request)
            results[model] = result.metrics
        
        # Create comparison summary
        comparison = {
            "benchmark_id": benchmark_id,
            "models": request.models,
            "benchmark_suite": request.benchmark_suite,
            "results": results,
            "winner": max(results.keys(), key=lambda k: results[k].get("accuracy", 0)),
            "created_at": datetime.utcnow().isoformat()
        }
        
        benchmarks_db[benchmark_id] = comparison
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error creating benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/evaluations/{evaluation_id}")
async def get_evaluation(evaluation_id: str):
    """Get evaluation results by ID"""
    if evaluation_id not in evaluations_db:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return evaluations_db[evaluation_id]

@app.get("/api/v1/evaluations")
async def list_evaluations():
    """List all evaluations"""
    return {
        "evaluations": list(evaluations_db.values()),
        "total": len(evaluations_db)
    }

@app.get("/api/v1/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """Get benchmark results by ID"""
    if benchmark_id not in benchmarks_db:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return benchmarks_db[benchmark_id]

@app.get("/api/v1/benchmarks")
async def list_benchmarks():
    """List all benchmarks"""
    return {
        "benchmarks": list(benchmarks_db.values()),
        "total": len(benchmarks_db)
    }

@app.get("/api/v1/test-cases/{dataset}")
async def get_test_cases(dataset: str):
    """Get test cases for a specific dataset"""
    test_cases = await generate_test_cases(dataset)
    return {"dataset": dataset, "test_cases": test_cases}

# Helper functions
async def generate_test_cases(dataset: str) -> List[TestCase]:
    """Generate test cases for evaluation"""
    # Simulate test case generation
    test_cases = []
    for i in range(10):  # Generate 10 test cases
        test_case = TestCase(
            id=str(uuid.uuid4()),
            input_data={
                "prompt": f"Test prompt {i+1} for {dataset}",
                "context": f"Context for test case {i+1}",
                "parameters": {"temperature": 0.7, "max_tokens": 100}
            },
            expected_output=f"Expected output for test case {i+1}",
            metadata={"dataset": dataset, "difficulty": "medium"}
        )
        test_cases.append(test_case)
    
    return test_cases

async def run_evaluation(model_name: str, test_cases: List[TestCase], parameters: Dict[str, Any]) -> List[TestCase]:
    """Run evaluation on test cases"""
    # Simulate model evaluation
    for test_case in test_cases:
        # Simulate model inference
        test_case.actual_output = f"Generated output for {test_case.input_data['prompt']}"
        test_case.score = 0.8 + (hash(test_case.id) % 20) / 100  # Simulate score between 0.8-1.0
    
    return test_cases

async def calculate_metrics(test_cases: List[TestCase], metrics: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    results = {}
    
    if "accuracy" in metrics:
        results["accuracy"] = sum(1 for tc in test_cases if tc.score and tc.score > 0.7) / len(test_cases)
    
    if "average_score" in metrics:
        results["average_score"] = sum(tc.score for tc in test_cases if tc.score) / len(test_cases)
    
    if "latency" in metrics:
        results["latency"] = 0.25  # Simulate 250ms average latency
    
    if "throughput" in metrics:
        results["throughput"] = 4.0  # Simulate 4 requests per second
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)