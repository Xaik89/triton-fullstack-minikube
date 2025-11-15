import numpy as np
import tritonclient.grpc as grpcclient
from typing import Dict, List, Tuple, Optional
import time


class TritonClient:
    def __init__(self, url: str, model_name: str, timeout: float = 60.0):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.client = None
        self._connect()

    def _connect(self):
        """Establish connection to Triton server"""
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=False
            )
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {self.url} is not live")
            if not self.client.is_model_ready(self.model_name):
                raise ConnectionError(f"Model {self.model_name} is not ready")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Triton: {str(e)}")

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: List[str],
        request_id: str = ""
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference on Triton server
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
            outputs: List of output names to retrieve
            request_id: Optional request ID for tracking
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if self.client is None:
            self._connect()

        # Prepare input tensors
        triton_inputs = []
        for name, data in inputs.items():
            triton_input = grpcclient.InferInput(name, data.shape, self._numpy_to_triton_dtype(data.dtype))
            triton_input.set_data_from_numpy(data)
            triton_inputs.append(triton_input)

        # Prepare output tensors
        triton_outputs = []
        for name in outputs:
            triton_output = grpcclient.InferRequestedOutput(name)
            triton_outputs.append(triton_output)

        # Perform inference
        start_time = time.time()
        response = self.client.infer(
            model_name=self.model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            request_id=request_id,
            headers={}
        )
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Extract results
        results = {}
        for name in outputs:
            results[name] = response.as_numpy(name)

        results['_inference_time_ms'] = inference_time
        return results

    def _numpy_to_triton_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to Triton dtype string"""
        dtype_map = {
            np.dtype('float32'): 'FP32',
            np.dtype('float16'): 'FP16',
            np.dtype('int32'): 'INT32',
            np.dtype('int64'): 'INT64',
            np.dtype('uint8'): 'UINT8',
        }
        return dtype_map.get(dtype, 'FP32')

    def is_ready(self) -> bool:
        """Check if model is ready"""
        try:
            if self.client is None:
                return False
            return self.client.is_model_ready(self.model_name)
        except:
            return False

    def close(self):
        """Close the client connection"""
        if self.client:
            self.client.close()
            self.client = None

