#!/bin/bash

echo "=== Sketch-to-Image Docker Setup ==="
echo "Setting up Docker environment and dependencies..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Get the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Create required directories
echo "Creating required directories..."
mkdir -p backend/dataset/sketch backend/dataset/result backend/dataset/metadata
chmod -R 755 backend/dataset
echo "✓ Created dataset directories with proper permissions"

# Detect GPU and configure environment
echo "Detecting hardware configuration..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected! Testing Docker GPU support..."
    
    if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null 2>&1; then
        echo "✓ NVIDIA Container Toolkit is working correctly"
        DEVICE_SETTING="cuda"
        echo "  GPU acceleration will be enabled"
    else
        echo "⚠ Warning: NVIDIA GPU detected but Docker GPU support not available"
        echo "  Install NVIDIA Container Toolkit for GPU acceleration"
        echo "  Falling back to CPU mode"
        DEVICE_SETTING="cpu"
    fi
else
    echo "No NVIDIA GPU detected - using CPU mode"
    DEVICE_SETTING="cpu"
fi

# Create environment configuration
echo "Creating environment configuration..."
cat > .env << EOF
# Device Configuration
DEVICE=$DEVICE_SETTING

# Backend Settings
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
OUTPUT_IMAGE_SIZE=512

# Frontend Settings  
REACT_APP_API_URL=http://localhost:8000/api
FRONTEND_URL=http://localhost:3000

# Dataset Directory (direct save)
DATASET_DIR=dataset

# Hugging Face Token for private models
# HUGGING_FACE_HUB_TOKEN=your_token_here
EOF

echo "✓ Environment configuration created"

echo ""
echo "===== Docker Setup Complete! ====="
echo ""
echo "Directory structure:"
echo "  backend/dataset/sketch/    - Sketch dataset"
echo "  backend/dataset/result/    - Result dataset"
echo "  backend/dataset/metadata/  - Generation metadata"
echo ""
echo "To start the application:"
echo "  ./scripts/run/start_docker.sh"
echo ""
echo "Configuration:"
echo "  - Device: $DEVICE_SETTING"
echo "  - Backend will run on: http://localhost:8000"
echo "  - Frontend will run on: http://localhost:3000"
echo ""