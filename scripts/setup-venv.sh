#!/bin/bash

set -e

echo "=== Setting up Python Virtual Environment for TensorRT Conversion ==="

VENV_DIR="/home/andreykh/my_apps/triton-locally/venv"
REQUIREMENTS_FILE="/home/andreykh/my_apps/triton-locally/triton-cv/requirements.txt"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install basic requirements for TensorRT conversion
echo ""
echo "Installing TensorRT conversion requirements..."
pip install --no-cache-dir \
    ultralytics \
    torch \
    torchvision \
    numpy \
    pillow \
    onnx

# Install TensorRT from NVIDIA PyPI
echo ""
echo "Installing TensorRT from NVIDIA PyPI..."
echo "Note: This may take several minutes. If it gets stuck, press Ctrl+C and try manual installation."
echo ""

# Try installing with timeout and no build isolation (faster)
if timeout 300 pip install --no-cache-dir --no-build-isolation nvidia-tensorrt --extra-index-url https://pypi.nvidia.com 2>&1 | tee /tmp/tensorrt_install.log; then
    echo "✓ TensorRT installed successfully"
elif timeout 300 pip install --no-cache-dir nvidia-tensorrt --extra-index-url https://pypi.nvidia.com 2>&1 | tee /tmp/tensorrt_install.log; then
    echo "✓ TensorRT installed successfully (without --no-build-isolation)"
else
    echo "⚠ TensorRT installation via NVIDIA PyPI failed or timed out"
    echo ""
    echo "Trying alternative installation methods..."
    echo ""
    
    # Try installing tensorrt-linux (pre-built wheel, faster)
    echo "Attempting to install pre-built TensorRT wheel..."
    if timeout 180 pip install --no-cache-dir tensorrt --extra-index-url https://pypi.nvidia.com 2>&1; then
        echo "✓ TensorRT installed successfully (pre-built wheel)"
    else
        echo "⚠ TensorRT installation failed"
        echo ""
        echo "Manual installation options:"
        echo ""
        echo "Option 1: Install with verbose output to see what's happening:"
        echo "  pip install -v nvidia-tensorrt --extra-index-url https://pypi.nvidia.com"
        echo ""
        echo "Option 2: Install from NVIDIA website (recommended if pip fails):"
        echo "  1. Visit: https://developer.nvidia.com/tensorrt"
        echo "  2. Download TensorRT for your CUDA version"
        echo "  3. Extract and install:"
        echo "     pip install python/tensorrt-*.whl"
        echo ""
        echo "Option 3: Use Docker container with TensorRT (easiest):"
        echo "  docker run --gpus all -it nvcr.io/nvidia/tritonserver:23.10-py3"
        echo ""
        echo "Option 4: Skip TensorRT for now and convert model in Docker container"
        echo ""
        echo "Checking if TensorRT is available anyway..."
    fi
fi

# Check if TensorRT is available
echo ""
echo "Verifying TensorRT installation..."
if python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" 2>/dev/null; then
    echo "✓ TensorRT is available and working"
else
    echo "⚠ TensorRT is not available"
    echo ""
    echo "You may need to:"
    echo "1. Install CUDA toolkit if not already installed"
    echo "2. Install TensorRT from NVIDIA's website"
    echo "3. Or use a Docker container with TensorRT pre-installed"
    echo ""
    echo "For now, you can try:"
    echo "  pip install nvidia-tensorrt --extra-index-url https://pypi.nvidia.com"
fi

echo ""
echo "=== Virtual Environment Setup Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To convert the model, run:"
echo "  cd ~/my_apps/triton-locally/scripts"
echo "  source ../venv/bin/activate"
echo "  ./convert-model.sh"

