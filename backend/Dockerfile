# Use CUDA 11.8 with Ubuntu 22.04 for better package support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]