# Setting Up GPU Support with Docker Desktop and WSL2

This guide provides detailed instructions for setting up NVIDIA GPU support for running the GitCoFL client in Docker containers using WSL2.

## Prerequisites

1. Windows 10/11 with WSL2 enabled
2. Docker Desktop installed and configured to use WSL2
3. NVIDIA GPU with compatible drivers installed on Windows

## Setup Steps

### 1. Install NVIDIA Drivers in Windows

1. Download and install the latest NVIDIA drivers for your GPU from the [NVIDIA Driver Downloads page](https://www.nvidia.com/Download/index.aspx)
2. Restart your computer after installation

### 2. Configure WSL2 for GPU Support

1. Open PowerShell as Administrator and update WSL:
   ```powershell
   wsl --update
   ```

2. Restart WSL:
   ```powershell
   wsl --shutdown
   ```

3. Start your WSL distribution and check if NVIDIA GPU is recognized:
   ```bash
   nvidia-smi
   ```
   If this command shows your GPU information, WSL2 GPU support is working correctly.

### 3. Install NVIDIA Container Toolkit in WSL2

1. Inside your WSL2 distribution, add the NVIDIA package repositories:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```

2. Install the NVIDIA Container Toolkit:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

### 4. Configure Docker Desktop

1. Open Docker Desktop
2. Go to Settings > General and ensure "Use WSL 2 based engine" is checked
3. Go to Settings > Resources > WSL Integration and ensure integration with your WSL distribution is enabled
4. Apply and restart Docker Desktop

### 5. Verify GPU Support in Docker

Run a test container to verify GPU access:
```bash
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this command displays your GPU information, Docker containers can now access your GPU.

### 6. Build and Run GitCoFL with GPU Support

1. Build the Docker image:
   ```bash
   cd /path/to/GitCoFL
   ./example/decentralized_fl/client/build-docker.sh
   ```

2. Run the container with GPU support:
   ```bash
   docker run --gpus all --env-file example/decentralized_fl/client/.env gitcofl-decentralized-client
   ```

## Troubleshooting

### No GPU Access in Container

1. Check if NVIDIA drivers are properly installed in Windows:
   - Open Device Manager and verify NVIDIA GPU is listed without errors

2. Verify WSL2 GPU access:
   ```bash
   nvidia-smi
   ```

3. Check Docker GPU configuration:
   ```bash
   docker info | grep -i nvidia
   ```

4. Restart Docker Desktop and WSL:
   ```powershell
   wsl --shutdown
   ```
   Then restart Docker Desktop

### Performance Considerations

- Ensure you're using the latest version of WSL2
- Update NVIDIA drivers regularly
- Monitor GPU usage with `nvidia-smi` during training to ensure the GPU is being utilized

## Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Desktop WSL2 GPU Support](https://docs.docker.com/desktop/wsl/use-nvidia-gpu/)
- [Microsoft WSL2 GPU Support Documentation](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
