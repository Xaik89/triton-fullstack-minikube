#!/bin/bash

set -e

echo "=== Docker Installation Script ==="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run this script as root. It will use sudo when needed."
   exit 1
fi

# Check if Docker is already installed
DOCKER_INSTALLED=false
if command -v docker &> /dev/null; then
    echo "Docker is already installed:"
    docker --version
    echo "Skipping Docker installation, proceeding to nvidia-container-toolkit setup..."
    echo ""
    DOCKER_INSTALLED=true
fi

if [ "$DOCKER_INSTALLED" = false ]; then
    echo "Installing Docker..."

    # Update package index
    sudo apt-get update

    # Install prerequisites
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
    fi

    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add current user to docker group (to run docker without sudo)
    sudo usermod -aG docker $USER

    # Start and enable Docker service
    sudo systemctl start docker
    sudo systemctl enable docker

    echo ""
    echo "✓ Docker installed successfully!"
    echo ""
    docker --version
fi

# Check for NVIDIA GPU and install nvidia-container-toolkit if available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected. Installing nvidia-container-toolkit for GPU support..."
    
    # Add NVIDIA package repositories for Ubuntu 24.04
    # Clean up any existing broken repository file
    sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Verify we're on Ubuntu 24.04
    . /etc/os-release
    if [ "$ID" != "ubuntu" ] || [ "$VERSION_ID" != "24.04" ]; then
        echo "⚠ This script is configured for Ubuntu 24.04 only."
        echo "Detected: $ID $VERSION_ID"
        echo "Please install nvidia-container-toolkit manually for your distribution."
        exit 1
    fi
    
    echo "Ubuntu 24.04 detected. Setting up nvidia-container-toolkit repository..."
    
    # Add GPG key (remove existing file first to avoid prompt)
    sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    # Use Ubuntu 22.04 repository (compatible with Ubuntu 24.04)
    echo "Using Ubuntu 22.04 repository (compatible with Ubuntu 24.04)..."
    if curl -fsSL "https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list" > /tmp/nvidia-repo.list 2>/dev/null; then
        if grep -q "^deb" /tmp/nvidia-repo.list 2>/dev/null; then
            ARCH=$(dpkg --print-architecture)
            cat /tmp/nvidia-repo.list | \
                sed "s#\$(ARCH)#$ARCH#g" | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                grep "^deb" | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            rm -f /tmp/nvidia-repo.list
            echo "✓ Repository configured successfully"
        else
            rm -f /tmp/nvidia-repo.list
            echo "⚠ Failed to parse repository file"
            exit 1
        fi
    else
        echo "⚠ Could not fetch repository configuration"
        echo "Please install nvidia-container-toolkit manually following:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        exit 1
    fi
    
    # Install nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use nvidia runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo ""
    echo "✓ NVIDIA Container Toolkit installed and configured!"
    echo ""
    echo "Testing GPU access in Docker..."
    if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &>/dev/null; then
        echo "✓ Docker GPU access confirmed!"
    else
        echo "⚠ Docker GPU access test failed. You may need to log out and log back in."
    fi
else
    echo ""
    echo "No NVIDIA GPU detected. Skipping nvidia-container-toolkit installation."
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "⚠ IMPORTANT: You need to log out and log back in (or restart your terminal)"
echo "   for the docker group changes to take effect."
echo ""
echo "After logging back in, verify Docker installation:"
echo "  docker run hello-world"
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "And test GPU access:"
    echo "  docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi"
fi

