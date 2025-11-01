import numpy as np
from src.attacks.pgd import pgd_attack
import matplotlib.pyplot as plt
def transferability_matrix(models, dataloader, device='cpu', max_batches=10):
    L = len(models)
    counts = np.zeros((L,L))
    totals = np.zeros((L,L))
    for i, src in enumerate(models):
        src.eval()
        for b, (X,y) in enumerate(dataloader):
            if b >= max_batches: break
            X, y = X.to(device), y.to(device)
            X_adv = pgd_attack(src, X, y, eps=8/255, alpha=2/255, steps=10)
            for j, tgt in enumerate(models):
                with __import__('torch').no_grad():
                    preds = tgt(X_adv).argmax(dim=1)
                    counts[i,j] += (preds != y).cpu().numpy().sum()
                    totals[i,j] += y.size(0)
    return counts / (totals + 1e-12)

def print_transferability_matrix(P, model_names):
    L = len(model_names)
    print("Transferability Matrix (rows: source, cols: target)")
    header = " " * 15 + "".join(f"{name:>15}" for name in model_names)
    print(header)
    for i in range(L):
        row = f"{model_names[i]:<15}" + "".join(f"{P[i,j]:15.4f}" for j in range(L))
        print(row)
def plot_transferability_matrix(P, model_names, title='Adversarial Transferability Matrix', save_path=None):
    plt.figure()
    plt.imshow(P, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Transferability Rate')
    plt.xticks(ticks=np.arange(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(model_names)), labels=model_names)
    plt.title(title)
    plt.xlabel('Target Model')
    plt.ylabel('Source Model')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()