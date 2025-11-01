import numpy as np
import torch
import matplotlib.pyplot as plt
def compute_input_gradients(model, dataloader, device='cpu', max_batches=8):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    grads = []
    for i, (x,y) in enumerate(dataloader):
        if i >= max_batches: break
        x, y = x.to(device).requires_grad_(True), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        model.zero_grad()
        loss.backward()
        g = x.grad.detach().view(x.size(0), -1).cpu().numpy()
        grads.append(g)
    if len(grads) == 0:
        return np.zeros((0,0))
    grads = np.concatenate(grads, axis=0)
    return grads

def gradient_subspace_similarity(models, dataloader, device='cpu', max_batches=8):
    Gvecs = []
    for m in models:
        g = compute_input_gradients(m, dataloader, device=device, max_batches=max_batches)
        Gvecs.append(g.mean(axis=0) if g.size else np.zeros(1))
    L = len(Gvecs)
    S = np.zeros((L,L))
    norms = [np.linalg.norm(v)+1e-12 for v in Gvecs]
    for i in range(L):
        for j in range(L):
            S[i,j] = np.dot(Gvecs[i], Gvecs[j]) / (norms[i]*norms[j])
    return S

def gradient_subspace_distance(models, dataloader, device='cpu', max_batches=8):
    Gvecs = []
    for m in models:
        g = compute_input_gradients(m, dataloader, device=device, max_batches=max_batches)
        Gvecs.append(g.mean(axis=0) if g.size else np.zeros(1))
    L = len(Gvecs)
    D = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            D[i,j] = np.linalg.norm(Gvecs[i] - Gvecs[j])
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