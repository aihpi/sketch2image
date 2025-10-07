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

# Check for docker-compose or docker compose
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Building and starting containers..."
echo "This may take several minutes on the first run as models are downloaded..."

# Build and start containers
$COMPOSE_CMD up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Application Started Successfully!"
    echo ""
    echo "Access the application:"
    echo "  - Main App: http://localhost:3000"
    echo ""
    echo "Architecture:"
    echo "  - Express proxy (port 3000): Serves React + proxies API"
    echo "  - Backend (internal): Not exposed, accessible via proxy"
    echo ""
    echo "Useful commands:"
    echo "  - View all logs:     $COMPOSE_CMD logs -f"
    echo "  - View backend logs: $COMPOSE_CMD logs -f backend"
    echo "  - View proxy logs:   $COMPOSE_CMD logs -f app"
    echo "  - Stop services:     $COMPOSE_CMD down"
    echo ""
    echo "Note: The first generation may take longer as models are downloaded."
else
    echo "❌ Error: Failed to start services."
    echo "Check logs with: $COMPOSE_CMD logs"
    exit 1
fi