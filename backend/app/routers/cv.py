from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import io
from typing import List
import os
import logging

from ..models.schemas import CVDetectionResponse, DetectionBox
from ..clients.triton_client import TritonClient

router = APIRouter(prefix="/cv", tags=["cv"])
logger = logging.getLogger(__name__)

# Initialize Triton client for CV model
TRITON_CV_URL = os.getenv("TRITON_CV_URL", "triton-cv:8001")
CV_MODEL_NAME = os.getenv("CV_MODEL_NAME", "yolo8n")

triton_cv_client = None


def get_triton_cv_client():
    """Get or create Triton CV client"""
    global triton_cv_client
    if triton_cv_client is None:
        triton_cv_client = TritonClient(TRITON_CV_URL, CV_MODEL_NAME)
    return triton_cv_client


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for YOLOv8n model
    - Resize to 384x384
    - Convert to RGB
    - Normalize to [0, 1]
    - Convert to CHW format
    """
    # Resize maintaining aspect ratio with padding
    image = image.convert("RGB")
    target_size = 384
    
    # Calculate scaling factor
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded_image = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    padded_image.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    
    # Convert to numpy array and normalize
    img_array = np.array(padded_image, dtype=np.float32) / 255.0
    
    # Convert HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def postprocess_yolo_output(output: np.ndarray, conf_threshold: float = 0.25) -> List[DetectionBox]:
    """
    Postprocess YOLO output to extract bounding boxes.
    Expected output shapes:
      - [1, N, C] (already in proposal-major order)
      - [1, C, N] (needs transpose) e.g., exported YOLOv8: [1, 8, 3024]
    Format per proposal: [x_center, y_center, width, height, conf, class_probs...]
    """
    if output.ndim != 3:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    # Ensure shape is [1, N, C]
    if output.shape[1] < output.shape[2]:
        output = np.transpose(output, (0, 2, 1))

    # Remove batch dimension -> [N, C]
    output = output[0]

    detections: List[DetectionBox] = []

    for detection in output:
        if detection.shape[0] < 6:
            continue

        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]
        
        if confidence < conf_threshold:
            continue
        
        class_probs = detection[5:]
        class_id = int(np.argmax(class_probs))
        class_conf = class_probs[class_id]
        
        # Final confidence is objectness * class probability
        final_conf = confidence * class_conf
        
        if final_conf < conf_threshold:
            continue
        
        # Convert to x1, y1, x2, y2 format
        x1 = (x_center - width / 2) * 384
        y1 = (y_center - height / 2) * 384
        x2 = (x_center + width / 2) * 384
        y2 = (y_center + height / 2) * 384
        
        detections.append(DetectionBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=float(final_conf),
            class_id=class_id
        ))
    
    return detections


@router.post("/detect", response_model=CVDetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in image using YOLOv8n model
    """
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        preprocessed = preprocess_image(image)
        
        # Get Triton client
        client = get_triton_cv_client()
        
        # Perform inference
        results = client.infer(
            inputs={"images": preprocessed.astype(np.float32)},
            outputs=["output0"]
        )
        
        inference_time = results.get("_inference_time_ms", 0.0)
        output = results["output0"]
        
        # Postprocess results
        detections = postprocess_yolo_output(output)
        
        return CVDetectionResponse(
            detections=detections,
            processing_time_ms=inference_time
        )
        
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/health")
async def health_check():
    """Check health of CV Triton instance"""
    try:
        client = get_triton_cv_client()
        is_ready = client.is_ready()
        return {"status": "ready" if is_ready else "not ready"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
