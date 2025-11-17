#!/bin/bash

set -e

echo "=== Building Docker Images ==="

# Get minikube docker environment
eval $(minikube docker-env)

# Build backend image
echo "Building backend image..."
cd ../backend
docker build -t triton-backend:latest .
cd ../scripts

# Build Triton CV image
echo "Building Triton CV image..."
cd ../triton-cv
docker build -t triton-cv:latest .
cd ../scripts

# Build Triton Florence image
echo "Building Triton Florence image..."
cd ../triton-florence
docker build -t triton-florence:latest .
cd ../scripts

echo ""
echo "=== Images Built Successfully ==="
echo "Images available in minikube:"
docker images | grep -E "triton-backend|triton-cv|triton-florence"

echo ""
echo "=== Next Steps ==="
echo "1. Apply Kubernetes manifests:"
echo "   kubectl apply -R -f ../k8s/"
echo ""
echo "2. Check pod status:"
echo "   kubectl get pods -n triton-inference"
echo ""
echo "3. Access services:"
echo "   Backend: minikube service backend -n triton-inference"
echo "   Grafana: minikube service grafana -n triton-inference"
echo "   Prometheus: minikube service prometheus -n triton-inference"

