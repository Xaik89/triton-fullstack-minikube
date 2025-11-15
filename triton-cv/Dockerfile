FROM nvcr.io/nvidia/tritonserver:23.10-py3

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

# Copy conversion script and run it to generate model
# Note: In production, you would pre-convert the model and copy it here
# For now, we'll create a placeholder that will be populated during build
COPY scripts/convert_yolo.py /scripts/convert_yolo.py
RUN chmod +x /scripts/convert_yolo.py

# Set working directory
WORKDIR /models

# Expose ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Run Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--metrics-port=8002"]

