FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model conversion
RUN pip install --no-cache-dir \
    ultralytics \
    torch \
    torchvision \
    onnx \
    numpy

# Create model repository directory
RUN mkdir -p /models/yolo8n/1

# Copy model conversion script
COPY scripts/ /scripts/

# Copy model config
COPY models/yolo8n/config.pbtxt /models/yolo8n/config.pbtxt

# Copy PyTorch model file
COPY models/yolo8n_384_dg_micro-mAP-52_12_08_2025.pt /tmp/model.pt

# Copy ONNX export script
COPY scripts/export_onnx.py /scripts/export_onnx.py
RUN chmod +x /scripts/export_onnx.py

# Generate ONNX model during build (doesn't require GPU)
RUN cd /tmp && \
    python3 -c "from ultralytics import YOLO; m=YOLO('model.pt'); m.export(format='onnx', imgsz=384, simplify=True, opset=12, dynamic=False, half=False)" && \
    (mv /tmp/yolo8n_384_dg_micro-mAP-52_12_08_2025.onnx /models/yolo8n/1/model.onnx 2>/dev/null || \
     mv /tmp/model.onnx /models/yolo8n/1/model.onnx) && \
    ls -lh /models/yolo8n/1/

# Set working directory
WORKDIR /models

# Expose ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Run Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--metrics-port=8002"]

