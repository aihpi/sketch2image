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

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✓ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✓ Frontend stopped"
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

# Start frontend
echo "Starting frontend server..."
cd "$PROJECT_DIR/frontend"
conda activate sketch2image-frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Application Started Successfully!"
echo ""
echo "Services:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000/api"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID