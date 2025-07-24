#!/bin/bash

echo "=== Sketch-to-Image Demonstrator Setup (Conda Version) ==="
echo "Setting up conda environments and dependencies..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH. Please install Miniconda/Anaconda first."
    exit 1
fi

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Create required directories
echo "Creating required directories..."
mkdir -p backend/uploads backend/outputs backend/preprocessed
chmod -R 755 backend/uploads backend/outputs backend/preprocessed
echo "✓ Created and set permissions for backend directories"

# Setup Backend Environment
echo ""
echo "=== Setting up Backend Environment ==="
echo "Creating conda environment: sketch2image-backend"

# Remove existing environment if it exists
# conda env remove -n sketch2image-backend -y 2>/dev/null || true

# Create new environment
conda create -n sketch2image-backend-2 python=3.10 -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to create backend conda environment"
    exit 1
fi

echo "✓ Created sketch2image-backend environment"

# Activate backend environment and install dependencies
echo "Installing backend dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend-2

# Install system-level dependencies that would be in the Dockerfile
echo "Installing system dependencies..."
# Note: These would normally be installed via apt in Docker, 
# but in conda we'll use conda packages where possible
conda install -y \
    requests \
    urllib3 \
    certifi

# Install PyTorch with CUDA 11.8 support (matching Dockerfile)
echo "Installing PyTorch with CUDA 11.8 support (matching Dockerfile)..."
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

# Install HuggingFace and Diffusers packages (exact versions from Dockerfile)
echo "Installing HuggingFace and Diffusers packages..."
pip install --no-cache-dir \
    huggingface_hub==0.19.4 \
    transformers==4.35.2 \
    diffusers==0.23.1 \
    accelerate==0.23.0

echo "✓ AI/ML packages installed"

# Install OpenCV and safetensors (from Dockerfile)
echo "Installing OpenCV and safetensors..."
pip install --no-cache-dir \
    opencv-python-headless \
    safetensors==0.3.2

echo "✓ OpenCV and safetensors installed"

# Install remaining requirements from requirements.txt
echo "Installing remaining requirements from requirements.txt..."
cd "$PROJECT_DIR/backend"
pip install --no-cache-dir -r requirements.txt

echo "✓ All requirements.txt dependencies installed"

echo "✓ Backend dependencies installed"

# Create backend .env file
echo "Creating backend configuration..."
cat > .env << EOF
DEVICE=$DEVICE_SETTING
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
OUTPUT_IMAGE_SIZE=512
FRONTEND_URL=http://localhost:3000
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
PREPROCESSED_DIR=preprocessed
EOF

echo "✓ Backend configuration created"

# Setup Frontend Environment
echo ""
echo "=== Setting up Frontend Environment ==="

# Deactivate current environment
conda deactivate

# Remove existing environment if it exists
conda env remove -n sketch2image-frontend -y 2>/dev/null || true

# Create frontend environment (based on working solution)
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

# Create startup scripts
echo ""
echo "=== Creating Startup Scripts ==="

# Create start script
cat > "$PROJECT_DIR/start_app.sh" << 'EOF'
#!/bin/bash

echo "Starting Sketch2Image Application..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend server..."
cd "$SCRIPT_DIR/backend"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start"
    exit 1
fi

# Start frontend
echo "Starting frontend server..."
cd "$SCRIPT_DIR/frontend"
conda activate sketch2image-frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "===== Application Started Successfully! ====="
echo "- Backend: http://localhost:8000/api"
echo "- Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF

chmod +x "$PROJECT_DIR/start_app.sh"

# Create individual service scripts
cat > "$PROJECT_DIR/start_backend.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend
python main.py
EOF

cat > "$PROJECT_DIR/start_frontend.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-frontend
npm start
EOF

chmod +x "$PROJECT_DIR/start_backend.sh"
chmod +x "$PROJECT_DIR/start_frontend.sh"

echo "✓ Startup scripts created"

# Create test script
cat > "$PROJECT_DIR/test_setup.sh" << 'EOF'
#!/bin/bash
echo "Testing setup..."

# Test backend environment
echo "Testing backend environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import diffusers, transformers; print('✓ All backend packages imported successfully')"

# Test frontend environment
echo "Testing frontend environment..."
conda activate sketch2image-frontend
node --version
npm --version

echo "✓ Setup test completed successfully"
EOF

chmod +x "$PROJECT_DIR/test_setup.sh"

echo ""
echo "===== Setup Complete! ====="
echo ""
echo "Created conda environments:"
echo "  - sketch2image-backend (Python 3.10 + PyTorch + AI libraries)"
echo "  - sketch2image-frontend (Node.js 18 + React)"
echo ""
echo "Available scripts:"
echo "  - ./start_app.sh        : Start both backend and frontend"
echo "  - ./start_backend.sh    : Start only backend"
echo "  - ./start_frontend.sh   : Start only frontend"
echo "  - ./test_setup.sh       : Test the installation"
echo ""
echo "To start the application:"
echo "  ./start_app.sh"
echo ""
echo "Note: The first image generation will take longer as models are downloaded."

conda deactivate