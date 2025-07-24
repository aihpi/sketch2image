#!/bin/bash
echo "=== Sketch-to-Image Docker Launch (GPU Mode) ==="

# Navigate to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Create .env file for Docker Compose
echo "▶️ Creating .env configuration file..."
cat > .env << EOF
# -- Hugging Face Token --
HUGGING_FACE_HUB_TOKEN=YOUR_HUGGING_FACE_TOKEN_HERE

# -- Backend Settings --
DEVICE=cuda
HOST=0.0.0.0
PORT=8000
DEBUG_MODE=true
DEFAULT_MODEL_ID=controlnet_scribble
FRONTEND_URL=http://localhost:3000

# -- Frontend Settings --
REACT_APP_API_URL=http://localhost:8000/api
EOF
echo "✔️ Configuration file '.env' created."

# Build and launch containers
echo "▶️ Building and starting containers... This may take a while on the first run."
docker-compose up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "✅===== Application Started Successfully! =====✅"
    echo ""
    echo "  - Frontend is running at: http://localhost:3000"
    echo "  - Backend API is available at: http://localhost:8000/docs"
    echo ""
    echo "  The backend is using your NVIDIA GPU."
    echo ""
    echo "  To see logs: docker-compose logs -f"
    echo "  To stop: ./scripts/stop_docker.sh"
    echo ""
else
    echo "❌ Error: Failed to start services. Check the logs with 'docker-compose logs -f'"
    exit 1
fi
