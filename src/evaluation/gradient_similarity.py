"""
Gradient similarity analysis for ensemble models.

This module computes gradient subspace similarities and distances between
different models in an ensemble.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Union


def compute_input_gradients(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    max_batches: int = 8
) -> np.ndarray:
    """
    Compute input gradients for a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        device: Device to run computation on. If None, uses same device as model.
        max_batches: Maximum number of batches to process
    
    Returns:
        np.ndarray: Gradient vectors of shape [n_samples, n_features]
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    grads = []
    
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True).requires_grad_(True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        model.zero_grad()
        loss.backward()
        g = x.grad.detach().view(x.size(0), -1).cpu().numpy()
        grads.append(g)
    
    if len(grads) == 0:
        return np.zeros((0, 0))
    
    grads = np.concatenate(grads, axis=0)
    return grads

def gradient_subspace_similarity(
    models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    max_batches: int = 8
) -> np.ndarray:
    """
    Compute gradient subspace similarity matrix for ensemble models.
    
    Args:
        models: List of PyTorch models
        dataloader: DataLoader for computing gradients
        device: Device to run computation on. If None, uses same device as first model.
        max_batches: Maximum number of batches to process
    
    Returns:
        np.ndarray: Similarity matrix of shape [n_models, n_models]
    """
    if device is None:
        device = next(models[0].parameters()).device
    else:
        device = torch.device(device)
    
    Gvecs = []
    for m in models:
        g = compute_input_gradients(m, dataloader, device=device, max_batches=max_batches)
        Gvecs.append(g.mean(axis=0) if g.size > 0 else np.zeros(1))
    
    L = len(Gvecs)
    S = np.zeros((L, L))
    norms = [np.linalg.norm(v) + 1e-12 for v in Gvecs]
    
    for i in range(L):
        for j in range(L):
            S[i, j] = np.dot(Gvecs[i], Gvecs[j]) / (norms[i] * norms[j])
    
    return S

def gradient_subspace_distance(
    models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    max_batches: int = 8
) -> np.ndarray:
    """
    Compute gradient subspace distance matrix for ensemble models.
    
    Args:
        models: List of PyTorch models
        dataloader: DataLoader for computing gradients
        device: Device to run computation on. If None, uses same device as first model.
        max_batches: Maximum number of batches to process
    
    Returns:
        np.ndarray: Distance matrix of shape [n_models, n_models]
    """
    if device is None:
        device = next(models[0].parameters()).device
    else:
        device = torch.device(device)
    
    Gvecs = []
    for m in models:
        g = compute_input_gradients(m, dataloader, device=device, max_batches=max_batches)
        Gvecs.append(g.mean(axis=0) if g.size > 0 else np.zeros(1))
    
    L = len(Gvecs)
    D = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            D[i, j] = np.linalg.norm(Gvecs[i] - Gvecs[j])
    
    return D
def plot_gradient_similarity_matrix(S, title='Gradient Similarity Matrix', save_path=None):
    plt.figure()
    plt.imshow(S, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(title)
    plt.xlabel('Model Index')
    plt.ylabel('Model Index')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()