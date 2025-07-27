#!/usr/bin/env python
# filepath: /home/chokchai-fa/cs_chula/thesis/GitCoFL/example/decentralized_fl/client/test_gpu.py

"""
Simple script to test if PyTorch can access the GPU.
This script will be available inside the Docker container.
"""

import torch
import sys

def test_gpu():
    """Test if PyTorch can access the GPU."""
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("GPU is NOT accessible to PyTorch!")
        print("Please check your NVIDIA drivers and CUDA installation.")
        sys.exit(1)
    
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    # Get the name of the first GPU
    if gpu_count > 0:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU 0 name: {device_name}")
        
        # Create a small tensor on GPU
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()
            z = x + y
            print("Successfully performed tensor operations on GPU")
            print(f"Result tensor: {z}")
            print("\nGPU TEST SUCCESSFUL: PyTorch can access the GPU!")
        except Exception as e:
            print(f"Failed to perform tensor operations on GPU: {e}")
            sys.exit(1)
    else:
        print("No GPUs found!")
        sys.exit(1)

if __name__ == "__main__":
    test_gpu()
