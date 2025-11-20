"""
Device management utilities for GPU/CPU compatibility.
"""

import torch
import os


def get_device(device=None, verbose=True):
    """
    Get the best available device (GPU if available, else CPU).
    
    Args:
        device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)
                If None, auto-detects the best available device.
        verbose: Whether to print device information.
    
    Returns:
        torch.device: The device to use.
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            if verbose:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            if verbose:
                print("CUDA not available. Using CPU.")
    else:
        device = device.lower()
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                if verbose:
                    print("Warning: CUDA requested but not available. Falling back to CPU.")
                device = 'cpu'
            elif ':' in device:
                # Check if specific GPU exists
                gpu_id = int(device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    if verbose:
                        print(f"Warning: GPU {gpu_id} not available. Using GPU 0.")
                    device = 'cuda:0'
    
    device_obj = torch.device(device)
    
    if device_obj.type == 'cuda':
        # Set memory management
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
    
    return device_obj


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0.0, 0.0

