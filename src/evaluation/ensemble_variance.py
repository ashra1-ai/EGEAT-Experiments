import numpy as np
import torch
from typing import List, Tuple, Union

def ensemble_variance(
    models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    max_batches: int = 6
) -> Tuple[List[int], List[float]]:
    """
    Compute ensemble variance as a function of ensemble size.
    
    Args:
        models: List of PyTorch models in the ensemble
        dataloader: DataLoader for evaluation data
        device: Device to run computation on. If None, uses same device as first model.
        max_batches: Maximum number of batches to process
    
    Returns:
        Tuple of (K_values, variances) where K_values is list of ensemble sizes
        and variances is list of corresponding variance values
    """
    if device is None:
        device = next(models[0].parameters()).device
    else:
        device = torch.device(device)
    
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    variances = []
    Kvals = list(range(1, len(models) + 1))
    
    for k in Kvals:
        batch_vars = []
        for i, (X, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            
            per_losses = []
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            for m in models[:k]:
                m.eval()
                with torch.no_grad():
                    l = ce(m(X), y).cpu().numpy()  # shape (batch,)
                    per_losses.append(l)
            
            per_losses = np.stack(per_losses, axis=0)  # (k, batch)
            batch_vars.append(np.mean(np.var(per_losses, axis=0)))
        
        variances.append(np.mean(batch_vars))
    
    return Kvals, variances
def plot_ensemble_variance(Kvals, variances, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(Kvals, variances, marker='o')
    plt.xlabel('Number of Models in Ensemble (K)')
    plt.ylabel('Average Variance of Per-sample Losses')
    plt.title('Ensemble Variance vs Number of Models')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()