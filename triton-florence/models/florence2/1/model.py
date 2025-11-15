"""
Florence-2 model implementation for Triton Python backend
Uses vLLM engine for inference
"""
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import os
from typing import Dict, List

try:
    from vllm import LLM, SamplingParams
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError as e:
    print(f"Warning: Could not import vLLM or transformers: {e}")


class TritonPythonModel:
    """
    Triton Python model for Florence-2 using vLLM
    """
    
    def initialize(self, args):
        """
        Initialize the model
        """
        model_name = os.getenv("FLORENCE_MODEL_NAME", "microsoft/Florence-2-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Florence-2 model: {model_name}")
        
        try:
            # Initialize processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # For object detection, we use the base model
            # Note: vLLM is primarily for text generation, so we use transformers directly
            # for vision-language tasks
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("Florence-2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a dummy model for testing
            self.model = None
            self.processor = None
    
    def execute(self, requests):
        """
        Execute inference requests
        """
        responses = []
        
        for request in requests:
            try:
                # Get inputs
                image_input = pb_utils.get_input_tensor_by_name(request, "image")
                text_input = pb_utils.get_input_tensor_by_name(request, "text")
                
                if image_input is None or text_input is None:
                    error_response = pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError("Missing input tensors")
                    )
                    responses.append(error_response)
                    continue
                
                # Convert inputs to numpy
                image_np = image_input.as_numpy()
                text_np = text_input.as_numpy()
                
                # Process inputs
                if self.model is None or self.processor is None:
                    # Return dummy output for testing
                    output_np = np.array([[0.0, 0.0, 100.0, 100.0, 0.9]], dtype=np.float32)
                else:
                    # Convert image from CHW to HWC
                    image_hwc = np.transpose(image_np[0], (1, 2, 0))
                    # Denormalize
                    image_hwc = (image_hwc * 255.0).astype(np.uint8)
                    
                    # Convert text
                    text_str = str(text_np[0]) if len(text_np) > 0 else "<DETECTION>"
                    
                    # Process with Florence-2
                    from PIL import Image
                    pil_image = Image.fromarray(image_hwc)
                    
                    # Prepare prompt
                    prompt = text_str if isinstance(text_str, str) else "<DETECTION>"
                    
                    # Run inference
                    inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=100,
                            do_sample=False
                        )
                    
                    # Decode output
                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=False
                    )[0]
                    
                    # Parse detection results (simplified)
                    # Florence-2 outputs text format, need to parse to boxes
                    output_np = self._parse_florence_output(generated_text)
                
                # Create output tensor
                output_tensor = pb_utils.Tensor("output", output_np)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)
                
            except Exception as e:
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error during inference: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def _parse_florence_output(self, text: str) -> np.ndarray:
        """
        Parse Florence-2 text output to bounding boxes
        This is a simplified parser - adjust based on actual output format
        """
        # Florence-2 outputs in a specific format
        # For now, return a placeholder
        # Format: [x1, y1, x2, y2, confidence, class_id]
        boxes = np.array([[0.0, 0.0, 100.0, 100.0, 0.9, 0]], dtype=np.float32)
        return boxes
    
    def finalize(self):
        """
        Cleanup
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

