# GitCoFL Decentralized Client Docker Setup

This Docker setup allows you to run the GitCoFL decentralized federated learning client in a containerized environment with GPU support.

## Quick Start

### 1. Build the Docker Image

From the project root directory:

```bash
cd /home/chokchai-fa/cs_chula/thesis/GitCoFL
docker build -f example/decentralized_fl/client/Dockerfile -t gitcofl-decentralized-client .
```

Or use the provided build script:

```bash
cd example/decentralized_fl/client
./build-docker.sh
```

### 2. Test GPU Support (Optional)

If you're using GPU, test that it works correctly:

```bash
./test-docker-gpu.sh
```

This will verify that the container can access your NVIDIA GPU.

### 3. Setup Environment Variables

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 4. Run the Container

Run a single client:

```bash
# Run without GPU
docker run --env-file .env -v $(pwd)/data:/app/client/data gitcofl-decentralized-client

# Run with GPU support
docker run --gpus all --env-file .env -v $(pwd)/data:/app/client/data gitcofl-decentralized-client
```

Or use the provided scripts:

```bash
# Run with GPU support
./run-docker-gpu.sh

# Run multiple clients with GPU support
./run-multi-clients-gpu.sh 3  # Runs 3 clients
```

## Multi-Client Setup

To run multiple clients for decentralized FL, create multiple environment files:

```bash
# Client 1
cp .env.example .env.client1
# Edit with SAMPLE_NO=1

# Client 2  
cp .env.example .env.client2
# Edit with SAMPLE_NO=2

# Run multiple clients
docker run --name fl-client1 --gpus all --env-file .env.client1 -d gitcofl-decentralized-client
docker run --name fl-client2 --gpus all --env-file .env.client2 -d gitcofl-decentralized-client
```

## Environment Variables

- `GIT_FL_REPO`: Git repository URL for federated learning coordination
- `GIT_ACCESS_TOKEN`: GitHub access token for repository access
- `GIT_EMAIL`: Git email for commits
- `SAMPLE_NO`: Client identifier (unique number for each client)

## Volume Mounts

- `/app/client/data`: Mount your local data directory to persist training data

## Viewing Logs

```bash
# View logs for a running container
docker logs fl-client1

# Follow logs in real-time
docker logs -f fl-client1
```

## Stopping Containers

```bash
# Stop a specific client
docker stop fl-client1

# Stop all running GitCoFL clients
docker stop $(docker ps -q --filter ancestor=gitcofl-decentralized-client)
```

## GPU Support

This Docker image includes GPU support using NVIDIA CUDA 11.8 and PyTorch with CUDA capabilities. To use GPUs with Docker Desktop and WSL:

1. **Prerequisites:**
   - Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your WSL instance
   - Install NVIDIA drivers in Windows that are compatible with WSL2
   - Enable GPU support in Docker Desktop settings

2. **Verify GPU Access:**
   ```bash
   # Check if NVIDIA GPU is accessible in WSL
   nvidia-smi
   
   # Test GPU access in a container
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Monitor GPU Usage:**
   ```bash
   # From WSL command line
   watch -n 1 nvidia-smi
   ```

4. **Run with the Helper Scripts:**
   ```bash
   # For a single client
   ./run-docker-gpu.sh
   
   # For multiple clients
   ./run-multi-clients-gpu.sh 3  # Runs 3 clients
   ```

For detailed setup instructions and troubleshooting, see the [GPU-WSL-Setup.md](GPU-WSL-Setup.md) guide.

For troubleshooting GPU issues with Docker Desktop and WSL, you can also refer to the [official documentation](https://docs.docker.com/desktop/wsl/use-nvidia-gpu/).
