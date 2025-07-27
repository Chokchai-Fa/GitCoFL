#!/bin/bash
# filepath: /home/chokchai-fa/cs_chula/thesis/GitCoFL/example/decentralized_fl/client/run-multi-clients-gpu.sh

# Check for command line arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <number_of_clients>"
  echo "Example: $0 3"
  exit 1
fi

NUM_CLIENTS=$1

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

# Stop any existing containers
echo "Stopping any existing GitCoFL clients..."
docker stop $(docker ps -q --filter ancestor=gitcofl-decentralized-client) 2>/dev/null || true

# Create environment files for each client
echo "Creating environment files for $NUM_CLIENTS clients..."
for i in $(seq 1 $NUM_CLIENTS); do
  if [ ! -f .env.example ]; then
    echo "Error: No .env.example file found."
    exit 1
  fi
  
  cp .env.example .env.client$i
  echo "Created .env.client$i - Please edit with appropriate values"
  echo "Setting SAMPLE_NO=$i in .env.client$i"
  sed -i "s/^SAMPLE_NO=.*/SAMPLE_NO=$i/" .env.client$i
done

# Prompt to continue
echo
echo "Please edit all .env.client* files with appropriate values before continuing."
read -p "Continue with launching containers? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  exit 1
fi

# Launch containers
echo "Launching $NUM_CLIENTS clients with GPU support..."
for i in $(seq 1 $NUM_CLIENTS); do
  echo "Starting client $i..."
  docker run --gpus all \
    --env-file .env.client$i \
    -v $(pwd)/data:/app/client/data \
    --name gitcofl-client$i \
    -d \
    gitcofl-decentralized-client
done

echo
echo "All clients launched successfully!"
echo "To view logs: docker logs gitcofl-client<number>"
echo "To stop all clients: docker stop \$(docker ps -q --filter ancestor=gitcofl-decentralized-client)"
