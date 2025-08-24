#!/bin/bash

echo "=== Sketch-to-Image Conda Setup ==="
echo "Setting up conda environments and dependencies..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH. Please install Miniconda/Anaconda first."
    exit 1
fi

# Get the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create required directories
echo "Creating required directories..."
mkdir -p "$PROJECT_DIR/backend/dataset/sketch" "$PROJECT_DIR/backend/dataset/result" "$PROJECT_DIR/backend/dataset/metadata"
chmod -R 755 "$PROJECT_DIR/backend/dataset"
echo "✓ Created dataset directories with proper permissions"

# Setup Backend Environment
echo ""
echo "=== Setting up Backend Environment ==="
echo "Creating conda environment: sketch2image-backend"

# Remove existing environment if it exists
conda env remove -n sketch2image-backend -y 2>/dev/null || true

# Create new environment
conda create -n sketch2image-backend python=3.10 -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to create backend conda environment"
    exit 1
fi

echo "✓ Created sketch2image-backend environment"

# Activate backend environment and install dependencies
echo "Installing backend dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend

# Install system-level dependencies
echo "Installing system dependencies..."
conda install -y requests urllib3 certifi

# Install PyTorch with appropriate version
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - installing CUDA 11.8 version"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
    DEVICE_SETTING="cuda"
else
    echo "No NVIDIA GPU detected - installing CPU version"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    DEVICE_SETTING="cpu"
fi

echo "✓ PyTorch installed"

# Install remaining requirements
echo "Installing remaining requirements..."
cd "$PROJECT_DIR/backend"
pip install --no-cache-dir -r requirements.txt

echo "✓ Backend dependencies installed"

# Create backend .env file
echo "Creating backend configuration..."
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
FRONTEND_URL=http://localhost:3000

# Dataset Directory (direct save)
DATASET_DIR=dataset

# Hugging Face Token for private models
# HUGGING_FACE_HUB_TOKEN=your_token_here
EOF

echo "✓ Backend configuration created"

# Setup Frontend Environment
echo ""
echo "=== Setting up Frontend Environment ==="

# Deactivate current environment
conda deactivate

# Remove existing environment if it exists
conda env remove -n sketch2image-frontend -y 2>/dev/null || true

# Create frontend environment
echo "Creating conda environment: sketch2image-frontend"
conda create -n sketch2image-frontend -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to create frontend conda environment"
    exit 1
fi

echo "✓ Created sketch2image-frontend environment"

# Activate frontend environment and install Node.js
echo "Installing Node.js from conda-forge..."
conda activate sketch2image-frontend
conda install conda-forge::nodejs -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Node.js"
    exit 1
fi

echo "✓ Node.js installed successfully"

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd "$PROJECT_DIR/frontend"
npm install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install frontend dependencies"
    exit 1
fi

echo "✓ Frontend dependencies installed"

# Create frontend .env file
echo "Creating frontend configuration..."
cat > .env << EOF
REACT_APP_API_URL=http://localhost:8000/api
EOF

echo "✓ Frontend configuration created"

conda deactivate

echo ""
echo "===== Conda Setup Complete! ====="
echo ""
echo "Created conda environments:"
echo "  - sketch2image-backend (Python 3.10 + PyTorch + AI libraries)"
echo "  - sketch2image-frontend (Node.js + React)"
echo ""
echo "Directory structure:"
echo "  backend/dataset/sketch/    - Sketch dataset"
echo "  backend/dataset/result/    - Result dataset"
echo "  backend/dataset/metadata/  - Generation metadata"
echo ""
echo "To start the application:"
echo "  ./scripts/run/start_conda.sh"
echo ""
echo "Configuration:"
echo "  - Device: $DEVICE_SETTING"
echo "  - Backend will run on: http://localhost:8000"  
echo "  - Frontend will run on: http://localhost:3000"
echo ""
echo "Note: The first image generation will take longer as models are downloaded."