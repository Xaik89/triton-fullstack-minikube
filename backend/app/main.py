from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import os

from .routers import cv
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


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "message": "Triton Inference Backend API",
        "endpoints": {
            "cv": "/cv/detect",
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
    
    return HealthResponse(
        status="healthy",
        backend=backend_status,
        triton_cv=triton_cv_status
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

