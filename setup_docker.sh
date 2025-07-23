#!/bin/bash

echo "=== Sketch-to-Image Demonstrator Setup ==="
echo "Setting up the environment and launching the application..."

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

mkdir -p backend/uploads backend/outputs backend/preprocessed
echo "Created required directories."

chmod -R 777 backend/uploads backend/outputs backend/preprocessed
echo "Set directory permissions."

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected! Enabling GPU acceleration..."
    
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo "NVIDIA Container Toolkit is working correctly."
        
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

echo "Building and starting containers..."
echo "This may take several minutes on the first run as models are downloaded..."

docker pull nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 &> /dev/null &
docker pull node:18-alpine &> /dev/null &
wait

docker-compose build && docker-compose up -d

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