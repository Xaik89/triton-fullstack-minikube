from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None


class CVDetectionResponse(BaseModel):
    detections: List[DetectionBox]
    processing_time_ms: float


class FlorenceDetectionResponse(BaseModel):
    detections: List[DetectionBox]
    processing_time_ms: float
    raw_output: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    triton_cv: str
    triton_florence: str

