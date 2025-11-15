from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from PIL import Image
import numpy as np
import io
from typing import List, Optional
import os

from ..models.schemas import FlorenceDetectionResponse, DetectionBox
from ..clients.triton_client import TritonClient

router = APIRouter(prefix="/florence", tags=["florence"])

# Initialize Triton client for Florence model
TRITON_FLORENCE_URL = os.getenv("TRITON_FLORENCE_URL", "triton-florence:8001")
FLORENCE_MODEL_NAME = os.getenv("FLORENCE_MODEL_NAME", "florence2")

triton_florence_client = None


def get_triton_florence_client():
    """Get or create Triton Florence client"""
    global triton_florence_client
    if triton_florence_client is None:
        triton_florence_client = TritonClient(TRITON_FLORENCE_URL, FLORENCE_MODEL_NAME)
    return triton_florence_client


def preprocess_image_florence(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for Florence-2 model
    - Resize to model input size
    - Convert to RGB
    - Normalize
    """
    # Florence-2 typically uses 224x224 or 384x384
    # Using 384x384 for better detection quality
    target_size = 384
    image = image.convert("RGB")
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_text(prompt: str) -> np.ndarray:
    """
    Preprocess text prompt for Florence-2
    TODO: Implement proper tokenization using Florence-2's tokenizer
    The actual implementation should use the model's tokenizer from transformers
    """
    # TODO: Replace with actual Florence-2 tokenizer
    # Example: tokenizer = AutoTokenizer.from_pretrained("microsoft/Florence-2-base")
    #          tokens = tokenizer(prompt, return_tensors="np")
    #          return tokens["input_ids"]
    
    # Temporary: return prompt length as placeholder
    # This will need to be replaced with actual tokenization
    prompt_encoded = np.array([len(prompt)], dtype=np.int32)
    return prompt_encoded


@router.post("/detect", response_model=FlorenceDetectionResponse)
async def detect_objects_florence(
    file: UploadFile = File(...),
    prompt: str = Form(default="<DETECTION>")
):
    """
    Detect objects in image using Florence-2 model
    prompt: Detection task prompt (default: "<DETECTION>")
    """
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        preprocessed_image = preprocess_image_florence(image)
        
        # Preprocess text prompt
        preprocessed_text = preprocess_text(prompt)
        
        # Get Triton client
        client = get_triton_florence_client()
        
        # Perform inference
        # Note: Actual input names depend on Florence-2 model configuration
        results = client.infer(
            inputs={
                "image": preprocessed_image,
                "text": preprocessed_text
            },
            outputs=["output"]
        )
        
        inference_time = results.get("_inference_time_ms", 0.0)
        output = results["output"]
        
        # Postprocess results
        # TODO: Implement proper Florence-2 output parsing
        # Florence-2 output format depends on the model configuration
        # The actual output may be text tokens that need to be decoded and parsed
        detections = []
        
        # TODO: Parse actual Florence-2 output format
        # The model output needs to be decoded and parsed based on the actual format
        # This is a placeholder that returns empty detections until proper parsing is implemented
        if isinstance(output, np.ndarray):
            # Parse output based on actual Florence-2 model output format
            # This will need to be implemented based on the actual model output structure
            pass  # Placeholder - implement actual parsing
        
        return FlorenceDetectionResponse(
            detections=detections,
            processing_time_ms=inference_time,
            raw_output={"shape": str(output.shape) if isinstance(output, np.ndarray) else str(type(output))}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/health")
async def health_check():
    """Check health of Florence Triton instance"""
    try:
        client = get_triton_florence_client()
        is_ready = client.is_ready()
        return {"status": "ready" if is_ready else "not ready"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

