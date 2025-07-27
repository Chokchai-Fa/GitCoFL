#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../../.."

# Build the Docker image using the client Dockerfile but with project root context
echo "Building GitCoFL Decentralized Client Docker image..."
docker build -f example/decentralized_fl/client/Dockerfile -t gitcofl-decentralized-client .

echo "Docker image 'gitcofl-decentralized-client' built successfully!"
echo ""
echo "To run the container with GPU support:"
echo "docker run --gpus all --env-file example/decentralized_fl/client/.env gitcofl-decentralized-client"
