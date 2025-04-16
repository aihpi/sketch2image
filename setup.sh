#!/bin/bash

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
mkdir -p backend/uploads backend/outputs frontend/public/

# Ensure frontend public directory has all required files
if [ ! -f "frontend/public/index.html" ]; then
    echo "Creating frontend public files..."
    # These might be missing due to git configs or other reasons
    touch frontend/public/favicon.ico
    touch frontend/public/logo192.png
    touch frontend/public/logo512.png
fi

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected! Enabling GPU support in docker-compose.yml"
    sed -i 's/# deploy:/deploy:/g' docker-compose.yml
    sed -i 's/#   resources:/  resources:/g' docker-compose.yml
    sed -i 's/#     reservations:/    reservations:/g' docker-compose.yml
    sed -i 's/#       devices:/      devices:/g' docker-compose.yml
    sed -i 's/#         - driver: nvidia/        - driver: nvidia/g' docker-compose.yml
    sed -i 's/#           count: 1/          count: 1/g' docker-compose.yml
    sed -i 's/#           capabilities: \[gpu\]/          capabilities: [gpu]/g' docker-compose.yml
    
    # Update environment variable for device
    sed -i 's/DEVICE=cpu/DEVICE=cuda/g' .env
else
    echo "No NVIDIA GPU detected. Using CPU mode."
fi

# Build and start the containers
echo "Building and starting containers..."
echo "This may take several minutes on the first run..."

# Front-end build check
echo "Building frontend container..."
if ! docker-compose build frontend; then
    echo "Error: Failed to build frontend container. Fixing TypeScript issues..."
    
    # Try to fix known issues
    echo "Updating Excalidraw version and fixing type issues..."
    sed -i 's/"@excalidraw\/excalidraw": "\^0.16.0"/"@excalidraw\/excalidraw": "0.15.2"/g' frontend/package.json
    
    # Try building again
    if ! docker-compose build frontend; then
        echo "Error: Still unable to build frontend. Please check the logs for more details."
        exit 1
    fi
fi

# Backend build check
echo "Building backend container..."
if ! docker-compose build backend; then
    echo "Error: Failed to build backend container. Please check the logs for more details."
    exit 1
fi

# Start the containers
echo "Starting containers..."
if ! docker-compose up -d; then
    echo "Error: Failed to start containers. Check the logs for details."
    echo "You can see the logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "Setup completed! The application is now running."
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000/api"
echo ""
echo "To view the logs, run: docker-compose logs -f"
echo "To stop the application, run: docker-compose down"