import numpy as np
import torch

def ensemble_variance(models, dataloader, device='cpu', max_batches=6):
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    variances = []
    Kvals = list(range(1, len(models)+1))
    for k in Kvals:
        batch_vars = []
        for i, (X,y) in enumerate(dataloader):
            if i >= max_batches: break
            per_losses = []
            X, y = X.to(device), y.to(device)
            for m in models[:k]:
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