#!/bin/bash

set -e

echo "=== Minikube Setup Script ==="

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Minikube is not installed. Please install it first:"
    echo "  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64"
    echo "  sudo install minikube-linux-amd64 /usr/local/bin/minikube"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo ""
    echo "Please install Docker first by running:"
    echo "  ./scripts/install-docker.sh"
    echo ""
    echo "Or install Docker manually:"
    echo "  https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &>/dev/null; then
    echo "❌ Docker daemon is not accessible!"
    echo ""
    
    # Check if user is in docker group
    if groups | grep -q docker; then
        echo "You are in the docker group, but your current session hasn't picked up the changes."
        echo ""
        echo "Try one of these solutions:"
        echo ""
        echo "Option 1: Activate docker group in current session (quick fix):"
        echo "  newgrp docker"
        echo "  Then run this script again"
        echo ""
        echo "Option 2: Log out and log back in (or restart terminal)"
        echo ""
        echo "Option 3: Start Docker service if it's not running:"
        echo "  sudo systemctl start docker"
        echo "  sudo systemctl enable docker"
    else
        echo "You are not in the docker group."
        echo ""
        echo "Please add yourself to the docker group:"
        echo "  sudo usermod -aG docker $USER"
        echo ""
        echo "Then log out and log back in (or restart terminal), or use:"
        echo "  newgrp docker"
    fi
    
    # Check if Docker service is running
    if systemctl is-active docker &>/dev/null || systemctl is-active --quiet docker 2>/dev/null; then
        echo ""
        echo "Note: Docker service appears to be running, but you may need to activate the docker group."
    else
        echo ""
        echo "Also, please start Docker service:"
        echo "  sudo systemctl start docker"
    fi
    
    exit 1
fi

# Check if minikube is already running
if minikube status &>/dev/null; then
    echo "Minikube cluster already exists."
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing minikube cluster..."
        minikube delete
    else
        echo "Using existing minikube cluster. Starting if stopped..."
        minikube start
        exit 0
    fi
fi

echo "Starting minikube cluster..."

# Check for NVIDIA GPU and adjust memory accordingly
if command -v nvidia-smi &> /dev/null; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "NVIDIA GPU detected with ${GPU_MEMORY}MB memory"
    echo "Using 4GB RAM for minikube to leave resources for GPU workloads"
    MEMORY=4096
else
    echo "No NVIDIA GPU detected. Using 4GB RAM."
    MEMORY=4096
fi

# Start minikube with GPU support
minikube start \
    --driver=docker \
    --memory=${MEMORY} \
    --cpus=4 \
    --disk-size=50g \
    --addons=metrics-server \
    --addons=ingress

# Enable GPU support (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Attempting to enable GPU support..."
    if minikube addons enable gpu-device-plugin 2>/dev/null; then
        echo "✓ GPU device plugin enabled"
    else
        echo "⚠ GPU device plugin addon may not be available with Docker driver."
        echo "  GPU access will be available through Docker's --gpus flag in deployments."
    fi
    
    # Verify Docker GPU access
    echo "Verifying Docker GPU access..."
    if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &>/dev/null; then
        echo "✓ Docker GPU access confirmed"
    else
        echo "⚠ Docker GPU access not configured. Install nvidia-container-toolkit:"
        echo "  ./scripts/install-docker.sh"
    fi
fi

# Enable ingress
echo "Enabling ingress addon..."
minikube addons enable ingress

# Wait for minikube to be ready
echo "Waiting for minikube to be ready..."
minikube status

# Show cluster info
echo ""
echo "=== Cluster Information ==="
kubectl cluster-info

# Show nodes
echo ""
echo "=== Nodes ==="
kubectl get nodes

echo ""
echo "=== Minikube Setup Complete ==="
echo "To access minikube dashboard: minikube dashboard"
echo "To access services: minikube service <service-name> -n triton-inference"

