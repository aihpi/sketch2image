#!/bin/bash
set -e # Exit immediately if a command fails

# === Configuration ===
# The target GitHub organization for the packages.
GITHUB_ORG="aihpi"
# The GitHub username you will use to log in.
# IMPORTANT: CHANGE THIS TO YOUR GITHUB USERNAME.
GITHUB_USER="parissashahabi"

# Image names
FRONTEND_IMAGE_NAME="sketch2image-frontend"
BACKEND_IMAGE_NAME="sketch2image-backend"

# The target platform for the cluster.
PLATFORM="linux/amd64"

# --- Pre-flight Check ---
if [[ "$GITHUB_USER" == "YOUR_USERNAME_HERE" ]]; then
    echo "❌ Error: Please edit this script ($0) and set your GITHUB_USER."
    exit 1
fi

# --- Main Script ---
# Navigate to the project root directory from the script's location
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "▶ (1/4) Logging into GitHub Container Registry (ghcr.io)..."
echo "    You will be prompted for your password. Use your Personal Access Token (PAT)."
docker login ghcr.io -u "$GITHUB_USER"
if [ $? -ne 0 ]; then
    echo "❌ Docker login failed. Ensure your PAT is correct and has 'write:packages' scope."
    exit 1
fi
echo "✔ Login successful."

echo "▶ (2/4) Setting up Docker Buildx builder..."
# This ensures you have a builder capable of multi-platform builds.
docker buildx create --use --name mybuilder &>/dev/null || docker buildx use mybuilder
echo "✔ Buildx is ready."

# Define the full image tags
FRONTEND_IMAGE_TAG="ghcr.io/$GITHUB_ORG/$FRONTEND_IMAGE_NAME:latest"
BACKEND_IMAGE_TAG="ghcr.io/$GITHUB_ORG/$BACKEND_IMAGE_NAME:latest"

echo "▶ (3/4) Building and pushing Backend image for $PLATFORM..."
echo "    Target: $BACKEND_IMAGE_TAG"
docker buildx build \
    --platform $PLATFORM \
    -t $BACKEND_IMAGE_TAG \
    --push \
    ./backend

echo "✔ Backend image pushed successfully."

echo "▶ (4/4) Building and pushing Frontend image for $PLATFORM..."
echo "    Target: $FRONTEND_IMAGE_TAG"
docker buildx build \
    --platform $PLATFORM \
    -t $FRONTEND_IMAGE_TAG \
    --push \
    ./frontend

echo "✔ Frontend image pushed successfully."

echo ""
echo "✅===== Success! =====✅"
echo "Both images have been built for $PLATFORM and pushed to the '$GITHUB_ORG' organization packages."
echo "You can view them at: https://github.com/orgs/$GITHUB_ORG/packages"