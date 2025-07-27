#!/bin/bash
# filepath: /home/chokchai-fa/cs_chula/thesis/GitCoFL/example/decentralized_fl/client/run-docker-gpu.sh

# Check if .env file exists, otherwise use example
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "No .env file found. Creating one from .env.example..."
    cp .env.example .env
    echo "Please edit the .env file with your actual values before running again."
    exit 1
  else
    echo "Error: No .env or .env.example file found."
    exit 1
  fi
fi

# Check if NVIDIA GPU is accessible
if ! command -v nvidia-smi &> /dev/null; then
  echo "Warning: nvidia-smi command not found. NVIDIA drivers might not be installed or configured properly."
  echo "See GPU-WSL-Setup.md for setup instructions."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
else
  echo "NVIDIA GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Create data directory if it doesn't exist
mkdir -p data

# Run the Docker container with GPU support
echo "Running GitCoFL Decentralized Client with GPU support..."
docker run --gpus all \
  --env-file .env \
  -v $(pwd)/data:/app/client/data \
  --name gitcofl-client \
  -it \
  gitcofl-decentralized-client

echo "Container exited. To view logs: docker logs gitcofl-client"
