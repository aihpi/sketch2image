#!/bin/bash

echo "=== Starting Sketch-to-Image (Docker) ==="

# Get the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Check if setup was run
if [ ! -f ".env" ]; then
    echo "Error: Environment not configured. Please run setup first:"
    echo "  ./scripts/setup/setup_docker.sh"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Building and starting containers..."
echo "This may take several minutes on the first run as models are downloaded..."

# Build and start containers
docker-compose up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Application Started Successfully!"
    echo ""
    echo "Services:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000/api"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo ""
    echo "Useful commands:"
    echo "  - View logs: docker-compose logs -f"
    echo ""
    echo "Note: The first generation may take longer as models are downloaded."
else
    echo "❌ Error: Failed to start services."
    echo "Check logs with: docker-compose logs"
    exit 1
fi