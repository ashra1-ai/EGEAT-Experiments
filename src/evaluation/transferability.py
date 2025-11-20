"""
Transferability analysis for adversarial examples.

This module computes transferability matrices showing how adversarial examples
generated from one model transfer to other models in the ensemble.
"""

import numpy as np
import torch
from typing import List, Union
from src.attacks.pgd import pgd_attack


def transferability_matrix(
    models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    max_batches: int = 10,
    epsilon: float = 8/255,
    alpha: float = 2/255,
    iters: int = 10
) -> np.ndarray:
    """
    Compute transferability matrix for ensemble models.
    
    The matrix P[i,j] represents the fraction of adversarial examples generated
    from model i that successfully fool model j.
    
    Args:
        models: List of PyTorch models in the ensemble
        dataloader: DataLoader for test data
        device: Device to run computation on. If None, uses same device as models.
        max_batches: Maximum number of batches to process
        epsilon: Perturbation magnitude for PGD attack
        alpha: Step size for PGD attack
        iters: Number of PGD iterations
    
    Returns:
        np.ndarray: Transferability matrix of shape [n_models, n_models]
    """
    if device is None:
        device = next(models[0].parameters()).device
    else:
        device = torch.device(device)
    
    L = len(models)
    counts = np.zeros((L, L))
    totals = np.zeros((L, L))
    
    for i, src_model in enumerate(models):
        src_model.eval()
        print(f"Generating adversarial examples from model {i+1}/{L}...")
        
        for b, (X, y) in enumerate(dataloader):
            if b >= max_batches:
                break
            
            X, y = X.to(device), y.to(device)
            
            # Generate adversarial examples using source model
            X_adv = pgd_attack(
                src_model, X, y,
                epsilon=epsilon,
                alpha=alpha,
                iters=iters,
                device=device
            )
            
            # Test transferability to all target models
            for j, tgt_model in enumerate(models):
                tgt_model.eval()
                with torch.no_grad():
                    preds = tgt_model(X_adv).argmax(dim=1)
                    incorrect = (preds != y).sum().item()
                    counts[i, j] += incorrect
                    totals[i, j] += y.size(0)
    
    # Compute transferability rates
    transferability = counts / (totals + 1e-12)
    
    return transferability

def print_transferability_matrix(P: np.ndarray, model_names: List[str] = None):
    """
    Print transferability matrix in a formatted table.
    
    Args:
        P: Transferability matrix
        model_names: Optional list of model names for display
    """
    L = len(P)
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(L)]
    
    print("\nTransferability Matrix (rows: source, cols: target)")
    print("=" * (15 + 15 * L))
    header = " " * 15 + "".join(f"{name:>15}" for name in model_names)
    print(header)
    print("-" * (15 + 15 * L))
    for i in range(L):
        row = f"{model_names[i]:<15}" + "".join(f"{P[i,j]:15.4f}" for j in range(L))
        print(row)
    print("=" * (15 + 15 * L))


def plot_transferability_matrix(
    P: np.ndarray,
    model_names: List[str] = None,
    title: str = 'Adversarial Transferability Matrix',
    save_path: str = None
):
    """
    Plot transferability matrix as a heatmap.
    
    Args:
        P: Transferability matrix
        model_names: Optional list of model names for display
        title: Plot title
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    L = len(P)
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(L)]
    
    plt.figure(figsize=(8, 7))
    plt.imshow(P, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Transferability Rate')
    plt.xticks(ticks=np.arange(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(model_names)), labels=model_names)
    plt.title(title)
    plt.xlabel('Target Model')
    plt.ylabel('Source Model')
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()