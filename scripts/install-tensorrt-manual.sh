#!/bin/bash

set -e

echo "=== Manual TensorRT Installation Guide ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./setup-venv.sh first"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Virtual environment activated"
echo ""

# Check CUDA version
echo "Checking CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "⚠ nvidia-smi not found"
    CUDA_VERSION="unknown"
fi

echo ""
echo "TensorRT Installation Options:"
echo ""
echo "1. Try installing with verbose output (recommended first):"
echo "   pip install -v --no-cache-dir nvidia-tensorrt --extra-index-url https://pypi.nvidia.com"
echo ""
echo "2. Try installing without build isolation (faster):"
echo "   pip install --no-cache-dir --no-build-isolation nvidia-tensorrt --extra-index-url https://pypi.nvidia.com"
echo ""
echo "3. Install from NVIDIA website:"
echo "   a) Visit: https://developer.nvidia.com/tensorrt"
echo "   b) Download TensorRT for CUDA $CUDA_VERSION"
echo "   c) Extract the archive"
echo "   d) Install Python package:"
echo "      pip install <extracted_path>/python/tensorrt-*.whl"
echo ""
echo "4. Use Docker (easiest, recommended):"
echo "   The Triton Docker image already has TensorRT installed"
echo "   You can convert the model inside the Docker container"
echo ""
echo "Which option would you like to try? (1-4, or 'q' to quit)"
read -p "> " choice

case $choice in
    1)
        echo "Installing with verbose output..."
        pip install -v --no-cache-dir nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
        ;;
    2)
        echo "Installing without build isolation..."
        pip install --no-cache-dir --no-build-isolation nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
        ;;
    3)
        echo "Please download TensorRT from: https://developer.nvidia.com/tensorrt"
        echo "Then extract and run: pip install <path>/python/tensorrt-*.whl"
        ;;
    4)
        echo "Using Docker is recommended. The conversion can be done in the Triton container."
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "Verifying TensorRT installation..."
if python3 -c "import tensorrt; print(f'✓ TensorRT version: {tensorrt.__version__}')" 2>/dev/null; then
    echo "✓ TensorRT is installed and working!"
else
    echo "⚠ TensorRT is not available"
fi

