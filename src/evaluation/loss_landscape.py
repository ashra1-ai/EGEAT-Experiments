import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def param_vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

def set_param_vector(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())

def loss_on_vector(model, vec, X, y, device='cpu'):
    set_param_vector(model, vec)
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = torch.nn.functional.cross_entropy(logits, y.to(device)).item()
    return loss

def scan_2d_loss(model, dataloader, device='cpu', grid_n=21, radius=1.0):
    Xs, ys = [], []
    for i, (X, y) in enumerate(dataloader):
        Xs.append(X); ys.append(y)
        if i >= 2: break
    X, y = torch.cat(Xs, dim=0), torch.cat(ys, dim=0)
    base = param_vector(model).to(device)
    D = base.numel()
    dir_a = torch.randn(D, device=device); dir_a /= dir_a.norm()
    dir_b = torch.randn(D, device=device); dir_b -= (dir_b.dot(dir_a))*dir_a; dir_b /= dir_b.norm()
    alphas = np.linspace(-radius, radius, grid_n)
    betas = np.linspace(-radius, radius, grid_n)
    losses = np.zeros((grid_n, grid_n))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vec = base + a*dir_a + b*dir_b
            losses[i,j] = loss_on_vector(model, vec, X, y, device=device)
    set_param_vector(model, base)
    return alphas, betas, losses

def scan_1d_loss(model, dataloader, device='cpu', grid_n=21, radius=1.0):
    Xs, ys = [], []
    for i, (X, y) in enumerate(dataloader):
        Xs.append(X); ys.append(y)
        if i >= 2: break
    X, y = torch.cat(Xs, dim=0), torch.cat(ys, dim=0)
    base = param_vector(model).to(device)
    D = base.numel()
    direction = torch.randn(D, device=device); direction /= direction.norm()
    alphas = np.linspace(-radius, radius, grid_n)
    losses = np.zeros(grid_n)
    for i, a in enumerate(alphas):
        vec = base + a*direction
        losses[i] = loss_on_vector(model, vec, X, y, device=device)
    set_param_vector(model, base)
    return alphas, losses
def plot_loss_landscape_2d(alphas, betas, losses, title='Loss Landscape', save_path=None):
    A, B = np.meshgrid(alphas, betas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, losses.T, cmap='viridis')
    ax.set_xlabel('Alpha Direction')
    ax.set_ylabel('Beta Direction')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()