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
