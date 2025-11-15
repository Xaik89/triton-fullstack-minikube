# Triton Inference Server on Minikube

This project demonstrates how to set up a complete machine learning inference pipeline using NVIDIA Triton Inference Server on Minikube (local Kubernetes). The system includes:

- **FastAPI Backend**: Routes requests to appropriate Triton instances
- **Triton CV Instance**: Serves YOLOv8n model (TensorRT optimized)
- **Triton Florence Instance**: Serves Florence-2 model using vLLM engine
- **Prometheus**: Collects metrics from all services
- **Grafana**: Visualizes metrics and provides dashboards

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Requests                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                    │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  /cv/detect  │              │/florence/detect│           │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          │ gRPC                         │ gRPC
          ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│  Triton CV Instance  │    │  Triton Florence Instance    │
│  (Port 8001)         │    │  (Port 8001)                 │
│                      │    │                              │
│  Model: YOLOv8n     │    │  Model: Florence-2           │
│  Backend: TensorRT   │    │  Backend: Python + vLLM      │
│  Metrics: 8002      │    │  Metrics: 8002               │
└──────────┬───────────┘    └──────────────┬───────────────┘
           │                                │
           └────────────┬───────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    Prometheus          │
           │    (Port 9090)         │
           │    Scrapes metrics     │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │    Grafana             │
           │    (Port 3000)         │
           │    Visualizes metrics  │
           └────────────────────────┘
```

## Prerequisites

- **Docker**: Docker Engine with GPU support (nvidia-container-toolkit)
- **Minikube**: Latest version installed
- **kubectl**: Kubernetes command-line tool
- **NVIDIA GPU** (optional but recommended): For model inference acceleration
- **CUDA Toolkit** (if using GPU): For TensorRT conversion
- **Python 3.10+**: For local development

## Setup Instructions

### 1. Install Docker

Docker is required for minikube to run. If you don't have Docker installed, use the provided script:

```bash
cd scripts
./install-docker.sh
```

This script will:
- Install Docker Engine
- Configure GPU support (nvidia-container-toolkit) if NVIDIA GPU is detected
- Add your user to the docker group

**Important**: After running the script, you need to **log out and log back in** (or restart your terminal) for the docker group changes to take effect.

Verify Docker installation:
```bash
docker run hello-world
```

If you have an NVIDIA GPU, verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

For manual Docker installation, see: https://docs.docker.com/engine/install/

### 2. Install Minikube

#### On Linux:
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

#### On macOS:
```bash
brew install minikube
```

#### On Windows:
Download from [minikube releases](https://github.com/kubernetes/minikube/releases)

### 3. Install kubectl

Follow the [official kubectl installation guide](https://kubernetes.io/docs/tasks/tools/).

### 4. Setup Minikube Cluster

Run the setup script:

```bash
cd scripts
./setup-minikube.sh
```

This script will:
- Check for Docker installation and running status
- Start minikube with appropriate resources (4GB RAM, 4 CPUs - optimized for GPU workloads)
- Enable metrics-server addon
- Enable ingress addon
- Enable GPU support if available

Alternatively, manually start minikube:

```bash
minikube start --driver=docker --memory=4096 --cpus=4 --disk-size=50g
minikube addons enable metrics-server
minikube addons enable ingress
```

**Note**: The script uses 4GB RAM to leave resources for GPU workloads. Adjust the `--memory` parameter if needed.

### 5. Build Docker Images

Build all Docker images and load them into minikube:

```bash
cd scripts
./build-images.sh
```

This will build:
- `triton-backend:latest` - FastAPI backend service
- `triton-cv:latest` - Triton server with YOLOv8n model
- `triton-florence:latest` - Triton server with Florence-2 model

**Note**: The Triton CV image includes a conversion script. For production, you should pre-convert the YOLOv8n model to TensorRT format and include it in the image.

### 6. Deploy to Kubernetes

Apply all Kubernetes manifests:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy backend
kubectl apply -f k8s/backend/

# Deploy Triton CV
kubectl apply -f k8s/triton-cv/

# Deploy Triton Florence
kubectl apply -f k8s/triton-florence/

# Deploy Prometheus
kubectl apply -f k8s/prometheus/

# Deploy Grafana
kubectl apply -f k8s/grafana/
```

Or apply everything at once:

```bash
kubectl apply -f k8s/
```

### 7. Verify Deployment

Check that all pods are running:

```bash
kubectl get pods -n triton-inference
```

Wait for all pods to be in `Running` state:

```bash
kubectl wait --for=condition=ready pod -l app=backend -n triton-inference --timeout=300s
kubectl wait --for=condition=ready pod -l app=triton-cv -n triton-inference --timeout=300s
kubectl wait --for=condition=ready pod -l app=triton-florence -n triton-inference --timeout=300s
```

### 8. Access Services

#### Backend API

```bash
minikube service backend -n triton-inference
```

Or port-forward:

```bash
kubectl port-forward -n triton-inference svc/backend 8000:8000
```

Then access at: `http://localhost:8000`

#### Grafana UI

```bash
minikube service grafana -n triton-inference
```

Or port-forward:

```bash
kubectl port-forward -n triton-inference svc/grafana 3000:3000
```

Then access at: `http://localhost:3000`
- Username: `admin`
- Password: `admin`

#### Prometheus UI

```bash
minikube service prometheus -n triton-inference
```

Or port-forward:

```bash
kubectl port-forward -n triton-inference svc/prometheus 9090:9090
```

Then access at: `http://localhost:9090`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### CV Object Detection (YOLOv8n)

Detect objects in an image:

```bash
curl -X POST "http://localhost:8000/cv/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "detections": [
    {
      "x1": 100.5,
      "y1": 150.2,
      "x2": 300.8,
      "y2": 400.5,
      "confidence": 0.95,
      "class_id": 0,
      "class_name": null
    }
  ],
  "processing_time_ms": 45.2
}
```

### Florence Object Detection

Detect objects using Florence-2:

```bash
curl -X POST "http://localhost:8000/florence/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg" \
  -F "prompt=<DETECTION>"
```

Response:
```json
{
  "detections": [
    {
      "x1": 100.5,
      "y1": 150.2,
      "x2": 300.8,
      "y2": 400.5,
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "object"
    }
  ],
  "processing_time_ms": 120.5,
  "raw_output": {
    "shape": "(1, 100)"
  }
}
```

## Models

### YOLOv8n (Computer Vision)

- **Model**: YOLOv8n from Ultralytics
- **Input**: RGB image, 640x640 pixels
- **Output**: Bounding boxes with confidence scores
- **Backend**: TensorRT (optimized for NVIDIA GPUs)
- **Conversion**: PyTorch → ONNX → TensorRT

### Florence-2 (Vision-Language)

- **Model**: Microsoft Florence-2-base
- **Input**: Image + text prompt
- **Output**: Object detections with labels
- **Backend**: Python backend with vLLM engine
- **Task**: Object detection via vision-language understanding

## Monitoring

### Prometheus Metrics

Prometheus automatically scrapes metrics from:
- Backend FastAPI service (port 8000)
- Triton CV instance (port 8002)
- Triton Florence instance (port 8002)

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `nv_inference_request_duration_us` - Triton inference time
- `nv_gpu_utilization` - GPU utilization
- `nv_inference_request_count` - Active inference requests

### Grafana Dashboards

Access the pre-configured dashboard at:
- URL: `http://localhost:3000`
- Dashboard: "Triton Inference Server Dashboard"

The dashboard includes:
- Request rate and latency
- Model inference time
- GPU utilization
- Error rates
- Active requests

## Troubleshooting

### Pods Not Starting

Check pod status:
```bash
kubectl get pods -n triton-inference
kubectl describe pod <pod-name> -n triton-inference
kubectl logs <pod-name> -n triton-inference
```

### GPU Not Available

If GPU is not available, the Triton instances will fall back to CPU (slower). To check GPU:

```bash
kubectl get nodes -o json | jq '.items[0].status.capacity'
```

### Model Not Loading

Check Triton server logs:
```bash
kubectl logs -f deployment/triton-cv -n triton-inference
kubectl logs -f deployment/triton-florence -n triton-inference
```

Verify model repository structure:
```bash
kubectl exec -it deployment/triton-cv -n triton-inference -- ls -la /models
```

### Connection Errors

Verify services are running:
```bash
kubectl get svc -n triton-inference
```

Test connectivity:
```bash
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://backend:8000/health
```

### Prometheus Not Scraping

Check Prometheus targets:
1. Access Prometheus UI: `http://localhost:9090`
2. Navigate to Status → Targets
3. Verify all targets are "UP"

### Grafana Not Showing Data

1. Verify Prometheus is configured as data source
2. Check Grafana logs: `kubectl logs deployment/grafana -n triton-inference`
3. Verify dashboard queries match available metrics

## Development

### Local Development

Run backend locally:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Model Conversion

Convert YOLOv8n to TensorRT:

```bash
cd triton-cv/scripts
python convert_yolo.py --model yolov8n.pt --output model.plan
```

### Updating Configurations

After modifying Kubernetes manifests:

```bash
kubectl apply -f k8s/
kubectl rollout restart deployment/<deployment-name> -n triton-inference
```

## Cleanup

Remove all resources:

```bash
kubectl delete namespace triton-inference
```

Stop minikube:

```bash
minikube stop
```

Delete minikube cluster:

```bash
minikube delete
```

## Project Structure

```
triton-locally/
├── backend/                 # FastAPI backend service
│   ├── app/
│   │   ├── main.py         # Main application
│   │   ├── routers/        # API routes
│   │   ├── models/         # Pydantic models
│   │   └── clients/        # Triton client
│   └── Dockerfile
├── triton-cv/              # Triton CV instance
│   ├── models/             # Model repository
│   ├── scripts/            # Conversion scripts
│   └── Dockerfile
├── triton-florence/        # Triton Florence instance
│   ├── models/             # Model repository
│   └── Dockerfile
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml
│   ├── backend/
│   ├── triton-cv/
│   ├── triton-florence/
│   ├── prometheus/
│   └── grafana/
├── monitoring/             # Monitoring configs
│   ├── prometheus/
│   └── grafana/
├── scripts/                # Setup scripts
│   ├── setup-minikube.sh
│   └── build-images.sh
└── README.md
```

## License

This project is provided as-is for demonstration purposes.

## References

- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Florence-2 Model](https://github.com/microsoft/Florence-2)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

