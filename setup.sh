#!/bin/bash

# Sketch-to-Image Demonstrator Setup Script
# This script sets up and runs the Sketch-to-Image application

echo "=== Sketch-to-Image Demonstrator Setup ==="
echo "Setting up the environment and launching the application..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p backend/uploads backend/outputs backend/preprocessed
echo "Created required directories."

# Set appropriate permissions
chmod -R 777 backend/uploads backend/outputs backend/preprocessed
echo "Set directory permissions."

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected! Enabling GPU acceleration..."
    
    # Test if NVIDIA Container Toolkit is installed
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo "NVIDIA Container Toolkit is working correctly."
        
        # Create temporary .env file with GPU settings
        cat > .env << EOF
DEVICE=cuda
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
OUTPUT_IMAGE_SIZE=512
FRONTEND_URL=http://localhost:3000
EOF
    else
        echo "Warning: NVIDIA GPU detected, but NVIDIA Container Toolkit is not properly configured."
        echo "Using CPU mode instead. For better performance, install NVIDIA Container Toolkit."
        
        # Create temporary .env file with CPU settings
        cat > .env << EOF
DEVICE=cpu
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
OUTPUT_IMAGE_SIZE=512
FRONTEND_URL=http://localhost:3000
EOF
    fi
else
    echo "No NVIDIA GPU detected. Using CPU mode."
    
    # Create temporary .env file with CPU settings
    cat > .env << EOF
DEVICE=cpu
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
OUTPUT_IMAGE_SIZE=512
FRONTEND_URL=http://localhost:3000
EOF
fi

echo "Environment configuration complete."

# Build and start the containers
echo "Building and starting containers..."
echo "This may take several minutes on the first run as models are downloaded..."

# Pull required Docker images
docker pull nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 &> /dev/null &
docker pull node:18-alpine &> /dev/null &
wait

# Build and start services
docker-compose build && docker-compose up -d

# Check if services started successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "===== Setup Complete! ====="
    echo "The Sketch-to-Image Demonstrator is now running."
    echo ""
    echo "- Frontend: http://localhost:3000"
    echo "- Backend API: http://localhost:8000/api"
    echo ""
    echo "The first generation may take longer as models are downloaded and initialized."
    echo ""
    echo "To stop the application: docker-compose down"
    echo "To view logs: docker-compose logs -f"
else
    echo "Error: Failed to start services. Please check the logs with: docker-compose logs"
    exit 1
fi