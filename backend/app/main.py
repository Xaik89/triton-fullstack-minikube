from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import os

from .routers import cv, florence
from .models.schemas import HealthResponse
from .clients.triton_client import TritonClient

app = FastAPI(
    title="Triton Inference Backend",
    description="FastAPI backend for serving models via Triton Inference Server",
    version="1.0.0"
)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(cv.router)
app.include_router(florence.router)


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "message": "Triton Inference Backend API",
        "endpoints": {
            "cv": "/cv/detect",
            "florence": "/florence/detect",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    backend_status = "healthy"
    
    # Check Triton CV
    triton_cv_url = os.getenv("TRITON_CV_URL", "triton-cv:8001")
    cv_model_name = os.getenv("CV_MODEL_NAME", "yolo8n")
    try:
        cv_client = TritonClient(triton_cv_url, cv_model_name)
        triton_cv_status = "ready" if cv_client.is_ready() else "not ready"
        cv_client.close()
    except:
        triton_cv_status = "unavailable"
    
    # Check Triton Florence
    triton_florence_url = os.getenv("TRITON_FLORENCE_URL", "triton-florence:8001")
    florence_model_name = os.getenv("FLORENCE_MODEL_NAME", "florence2")
    try:
        florence_client = TritonClient(triton_florence_url, florence_model_name)
        triton_florence_status = "ready" if florence_client.is_ready() else "not ready"
        florence_client.close()
    except:
        triton_florence_status = "unavailable"
    
    return HealthResponse(
        status="healthy",
        backend=backend_status,
        triton_cv=triton_cv_status,
        triton_florence=triton_florence_status
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

