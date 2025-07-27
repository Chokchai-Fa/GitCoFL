#!/bin/bash
# filepath: /home/chokchai-fa/cs_chula/thesis/GitCoFL/example/decentralized_fl/client/test-docker-gpu.sh

# Check if Docker image exists
if ! docker image inspect gitcofl-decentralized-client >/dev/null 2>&1; then
  echo "Error: Docker image 'gitcofl-decentralized-client' not found!"
  echo "Please build the image first using build-docker.sh"
  exit 1
fi

# Check if NVIDIA GPU is accessible through docker
echo "Testing if Docker can access NVIDIA GPU..."
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "Error: Docker cannot access NVIDIA GPU!"
  echo "Please check your GPU setup. See GPU-WSL-Setup.md for instructions."
  exit 1
fi

echo "Running GPU test in container..."
docker run --rm --gpus all gitcofl-decentralized-client python test_gpu.py

# Check exit status
if [ $? -eq 0 ]; then
  echo
  echo "Success! Your container can access the GPU."
  echo "You can now run the client with GPU support using run-docker-gpu.sh"
else
  echo
  echo "Failed to access GPU in the container."
  echo "Please check your setup. See GPU-WSL-Setup.md for troubleshooting."
fi
