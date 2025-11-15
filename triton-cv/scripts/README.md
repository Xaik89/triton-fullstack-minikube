# YOLOv8n Conversion Script

This script converts YOLOv8n model to TensorRT format for Triton Inference Server.

## Usage

```bash
python convert_yolo.py --model yolov8n.pt --output model.plan
```

The script will:
1. Download YOLOv8n if not present
2. Export to ONNX format
3. Convert ONNX to TensorRT engine (.plan)

## Requirements

- NVIDIA GPU with TensorRT support
- CUDA toolkit
- TensorRT library
- ultralytics package
- torch

## Note

For production use, convert the model during Docker image build or use a pre-converted model.

