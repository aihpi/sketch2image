FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent tzdata from prompting for input
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set working directory
WORKDIR /app

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    # Add OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python and pip to Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + CUDA-compatible packages
# Upgraded PyTorch to 2.0.0 which has better compatibility with newer diffusers
RUN pip install --no-cache-dir torch==2.0.0+cu117 torchvision==0.15.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install specific versions of HuggingFace packages for better compatibility
RUN pip install --no-cache-dir \
    huggingface_hub==0.19.4 \
    transformers==4.35.2 \
    diffusers==0.23.1 \
    accelerate==0.23.0

# Install memory optimization libraries
RUN pip install --no-cache-dir xformers

# Install dependencies for specific models and processors
RUN pip install --no-cache-dir \
    opencv-python-headless \
    safetensors==0.3.2

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]