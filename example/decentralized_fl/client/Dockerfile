# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for better Python experience
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-wheel \
    build-essential \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the GitCoFL package and setup files from project root
COPY gitcofl/ /app/gitcofl/
COPY setup.py /app/
COPY requirements.txt /app/
COPY README.md /app/

# Copy client-specific files
COPY example/decentralized_fl/client/ /app/client/
WORKDIR /app/client

# Make test script executable
RUN chmod +x /app/client/test_gpu.py

# Install the gitcofl package in development mode
RUN pip install -e /app

# Install client-specific dependencies with GPU support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Create data directory
RUN mkdir -p data

# Run the client
CMD ["python", "main.py"]