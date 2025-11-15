#!/bin/bash

set -e

echo "=== Converting YOLOv8n Model to TensorRT using Docker ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="$HOME/data/DG/models/yolo8n_384_dg_micro-mAP-52_12_08_2025.pt"
OUTPUT_DIR="$PROJECT_DIR/triton-cv/models/yolo8n/1"
OUTPUT_FILE="$OUTPUT_DIR/model.plan"
ONNX_FILE="$HOME/data/DG/models/yolo8n_384_dg_micro-mAP-52_12_08_2025.onnx"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    exit 1
fi

# Check if Docker can access GPU
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &>/dev/null; then
    echo "❌ Docker cannot access GPU!"
    echo "Please ensure nvidia-container-toolkit is installed and Docker is restarted"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy conversion script to a temp location that Docker can access
CONVERSION_SCRIPT="/tmp/convert_yolo_docker.py"
cat > "$CONVERSION_SCRIPT" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Convert YOLOv8n ONNX model to TensorRT engine for Triton Inference Server
"""
import os
import sys
import tensorrt as trt
import numpy as np

def onnx_to_tensorrt(onnx_path: str, engine_path: str, input_shape: tuple = (1, 3, 384, 384)):
    """
    Convert ONNX model to TensorRT engine
    """
    print(f"Converting ONNX to TensorRT engine...")
    print(f"ONNX: {onnx_path}")
    print(f"Output: {engine_path}")
    print(f"Input shape: {input_shape}")
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print(f"Network has {network.num_layers} layers")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Build engine
    print("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False
    
    # Save engine
    print(f"Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {engine_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_yolo_docker.py <onnx_path> <output_path> <input_h> <input_w>")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    output_path = sys.argv[2]
    input_h = int(sys.argv[3])
    input_w = int(sys.argv[4])
    
    input_shape = (1, 3, input_h, input_w)
    success = onnx_to_tensorrt(onnx_path, output_path, input_shape)
    
    if not success:
        sys.exit(1)
PYTHON_SCRIPT

chmod +x "$CONVERSION_SCRIPT"

# Check if ONNX file exists, if not, export it first
if [ ! -f "$ONNX_FILE" ]; then
    echo "ONNX file not found. Exporting model to ONNX first..."
    echo "This will be done in Docker container..."
    
    # Export to ONNX in Docker
    docker run --rm --gpus all \
        -v "$HOME/data/DG/models:/models" \
        -v "$PROJECT_DIR:/workspace" \
        nvcr.io/nvidia/tritonserver:23.10-py3 \
        python3 -c "
from ultralytics import YOLO
import os
model = YOLO('/models/yolo8n_384_dg_micro-mAP-52_12_08_2025.pt')
model.export(format='onnx', imgsz=384, simplify=True, opset=12, dynamic=False, half=False)
print('ONNX export complete')
"
    
    if [ ! -f "$ONNX_FILE" ]; then
        echo "❌ ONNX export failed!"
        exit 1
    fi
    echo "✓ ONNX export complete"
fi

echo "Converting ONNX to TensorRT using Docker..."
echo ""

# Run conversion in Docker container
docker run --rm --gpus all \
    -v "$HOME/data/DG/models:/models" \
    -v "$OUTPUT_DIR:/output" \
    -v "$CONVERSION_SCRIPT:/convert.py" \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    python3 /convert.py /models/yolo8n_384_dg_micro-mAP-52_12_08_2025.onnx /output/model.plan 384 384

# Clean up temp script
rm -f "$CONVERSION_SCRIPT"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Model converted successfully!"
    echo "  Location: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo ""
    echo "❌ Conversion failed!"
    exit 1
fi

