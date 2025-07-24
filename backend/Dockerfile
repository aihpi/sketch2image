# backend/Dockerfile
# This version is optimized for building and running on a GPU machine (like your laptop).

# Use the official CUDA 11.8 image.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# CORRECTED: Upgrade pip using python3.10 directly, THEN create the symlink.
# This is the fix for the "/bin/sh: 1: python: not found" error.
RUN pip install --upgrade pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements file
COPY requirements.txt .

# Install the GPU-version of PyTorch.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
