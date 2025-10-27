#!/bin/bash

echo "=== Starting Sketch-to-Image (Conda) ==="

# Get the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Check if conda environments exist
if ! conda env list | grep -q "sketch2image-backend"; then
    echo "Error: Backend environment not found. Please run setup first:"
    echo "  ./scripts/setup/setup_conda.sh"
    exit 1
fi

if ! conda env list | grep -q "sketch2image-frontend"; then
    echo "Error: Frontend environment not found. Please run setup first:"
    echo "  ./scripts/setup/setup_conda.sh"
    exit 1
fi

# Check if frontend build exists
if [ ! -d "$PROJECT_DIR/frontend/build" ]; then
    echo "Error: Frontend build not found. Please run setup first:"
    echo "  ./scripts/setup/setup_conda.sh"
    exit 1
fi

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✓ Backend stopped"
    fi
    if [ ! -z "$PROXY_PID" ]; then
        kill $PROXY_PID 2>/dev/null
        echo "✓ Proxy server stopped"
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend server..."
cd "$PROJECT_DIR/backend"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sketch2image-backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Error: Backend failed to start"
    exit 1
fi

echo "✓ Backend started successfully"

# Copy frontend build to proxy server
echo "Preparing proxy server..."
cp -r "$PROJECT_DIR/frontend/build" "$PROJECT_DIR/proxy-server/"

# Start proxy server
echo "Starting proxy server (Express + React)..."
cd "$PROJECT_DIR/proxy-server"
conda activate sketch2image-frontend
export BACKEND_URL=http://localhost:8000
npm start &
PROXY_PID=$!

# Wait for proxy to start
echo "Waiting for proxy server to initialize..."
sleep 3

# Check if proxy is running
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "❌ Error: Proxy server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✓ Proxy server started successfully"

echo ""
echo "✅ Application Started Successfully!"
echo ""
echo "Access the application:"
echo "  - Main App: http://localhost:3000"
echo ""
echo "Architecture:"
echo "  - Express proxy (port 3000): Serves React + proxies API"
echo "  - Backend (port 8000): Internal API server"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for processes
wait $BACKEND_PID $PROXY_PID