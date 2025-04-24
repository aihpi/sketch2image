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
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python and pip to Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + CUDA-compatible packages
RUN pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install HuggingFace + Diffusers + Accelerate
RUN pip install --no-cache-dir huggingface_hub==0.30.2 \
    transformers>=4.46.3 \
    diffusers>=0.33.0 \
    accelerate>=1.6.0

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install xformers==0.0.23
RUN pip install hf_xet

# Copy application code
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
