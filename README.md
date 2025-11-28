# Triton Inference Server on Minikube

This project demonstrates how to set up a complete machine learning inference pipeline using NVIDIA Triton Inference Server on Minikube (local Kubernetes). The system includes:

- **FastAPI Backend**: Routes requests to Triton CV instance
- **Triton CV Instance**: Serves YOLOv8n model (TensorRT optimized)
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
│  ┌──────────────┐                                            │
│  │  /cv/detect  │                                            │
│  └──────┬───────┘                                            │
└─────────┼────────────────────────────────────────────────────┘
          │
          │ gRPC
          ▼
┌──────────────────────┐
│  Triton CV Instance  │
│  (Port 8001)         │
│                      │
│  Model: YOLOv8n     │
│  Backend: TensorRT   │
│  Metrics: 8002      │
└──────────┬───────────┘
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

- **Docker** with GPU support (`nvidia-container-toolkit`)
- **Minikube**
- **kubectl**
- **NVIDIA GPU** (required for default manifests: `triton-cv` requests `nvidia.com/gpu`)
- **Python 3.10+** (for local tools / scripts)

## Quickstart

### 1. Install Docker (with GPU)

On Ubuntu, the easiest path is:

```bash
cd scripts
./install-docker.sh
```

This installs Docker, sets up NVIDIA Container Toolkit (if a GPU is present), and adds your user to the `docker` group.

Verify:

```bash
docker run hello-world
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

For other OSes, follow the official docs: https://docs.docker.com/engine/install/

### 2. Install Minikube and kubectl

- Minikube: https://minikube.sigs.k8s.io/docs/start/
- kubectl: https://kubernetes.io/docs/tasks/tools/

### 3. Start Minikube (with GPU)

Use the helper script (Docker driver):

```bash
cd scripts
./setup-minikube.sh
```

This script:
- Verifies Docker is installed and running
- Starts Minikube with 4 CPU / 4GB RAM
- Enables `metrics-server` and `ingress`
- Enables GPU support for the Docker driver (so the node exposes `nvidia.com/gpu`)

### 4. Build Docker images

```bash
cd scripts
./build-images.sh
```

Images built into the Minikube Docker:
- `triton-backend:latest`
- `triton-cv:latest`

### 5. Deploy to Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -R -f k8s/
```

Check pod status:

```bash
kubectl get pods -n triton-inference
```

### 6. Access services

- Backend:
  - `minikube service backend -n triton-inference`
  - or `kubectl port-forward -n triton-inference svc/backend 8000:8000`
- Grafana:
  - `minikube service grafana -n triton-inference`
  - or `kubectl port-forward -n triton-inference svc/grafana 3000:3000`
- Prometheus:
  - `minikube service prometheus -n triton-inference`
  - or `kubectl port-forward -n triton-inference svc/prometheus 9090:9090`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### CV Object Detection (YOLOv8n)

```bash
curl -X POST "http://localhost:8000/cv/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

Returns detection boxes and processing time.

## Models

### YOLOv8n (Computer Vision)

- **Model**: YOLOv8n from Ultralytics
- **Input**: RGB image, 640x640 pixels
- **Output**: Bounding boxes with confidence scores
- **Backend**: TensorRT (optimized for NVIDIA GPUs)
- **Conversion**: PyTorch → ONNX → TensorRT

## Monitoring

### Prometheus Metrics

Prometheus automatically scrapes metrics from:
- Backend FastAPI service (port 8000)
- Triton CV instance (port 8002)

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
triton-fullstack-minikube/
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
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml
│   ├── backend/
│   ├── triton-cv/
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
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
