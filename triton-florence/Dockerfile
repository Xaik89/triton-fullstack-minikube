FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    vllm==0.2.5 \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    pillow>=9.0.0 \
    numpy>=1.24.0 \
    accelerate

# Create model repository directory
RUN mkdir -p /models/florence2/1

# Copy model files
COPY models/florence2/config.pbtxt /models/florence2/config.pbtxt
COPY models/florence2/1/model.py /models/florence2/1/model.py

# Set environment variables
ENV FLORENCE_MODEL_NAME=microsoft/Florence-2-base
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /models

# Expose ports
# 8000: HTTP
# 8001: gRPC
# 8002: Metrics
EXPOSE 8000 8001 8002

# Run Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--metrics-port=8002"]

