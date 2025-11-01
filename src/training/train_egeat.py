import torch
import torch.nn.functional as F
from copy import deepcopy

def train_egeat_epoch(model, dataloader, optimizer, device='cuda',
                      lambda_geom=0.1, lambda_soup=0.05, epsilon=8/255,
                      ensemble_snapshots=None):
    """
    Train one epoch using EGEAT.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        lambda_geom: geometric regularization weight
        lambda_soup: ensemble smoothing weight
        epsilon: adversarial perturbation radius (L_inf)
        ensemble_snapshots: list of parameter snapshots (optional)
    """
    model.train()
    
    # Maintain running parameter centroid for ensemble smoothing
    if ensemble_snapshots is not None and len(ensemble_snapshots) > 0:
        theta_soup = deepcopy(ensemble_snapshots[0])
        for p in theta_soup.keys():
            for snap in ensemble_snapshots[1:]:
                theta_soup[p] += snap[p]
            theta_soup[p] /= len(ensemble_snapshots)
    else:
        theta_soup = None  # skip soup term if no snapshots
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        
        # -----------------------------
        # 1. Exact Input-Space Perturbation
        # -----------------------------
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        grad_x = torch.autograd.grad(loss, x, retain_graph=True, create_graph=False)[0]
        
        # L_inf dual norm: just sign of gradient
        delta_star = epsilon * grad_x.sign()
        x_adv = torch.clamp(x + delta_star, 0.0, 1.0)
        
        # -----------------------------
        # 2. Adversarial Loss
        # -----------------------------
        logits_adv = model(x_adv)
        L_adv = F.cross_entropy(logits_adv, y)
        
        # -----------------------------
        # 3. Geometric Regularization (Gradient Subspace decorrelation)
        # -----------------------------
        L_geom = 0.0
        if ensemble_snapshots is not None and len(ensemble_snapshots) > 0:
            # Example: cosine similarity between current grad and snapshot grads
            grad_vec = grad_x.view(grad_x.size(0), -1)  # flatten batch
            for snap in ensemble_snapshots:
                # compute model output on current x
                model.load_state_dict(snap)
                logits_snap = model(x)
                loss_snap = F.cross_entropy(logits_snap, y)
                grad_snap = torch.autograd.grad(loss_snap, x, retain_graph=True)[0].view(grad_x.size(0), -1)
                
                cos_sim = F.cosine_similarity(grad_vec, grad_snap, dim=1).mean()
                L_geom += cos_sim
            L_geom /= len(ensemble_snapshots)
        
        # -----------------------------
        # 4. Ensemble Smoothing Loss
        # -----------------------------
        L_soup = 0.0
        if theta_soup is not None:
            for p, p_soup in zip(model.parameters(), theta_soup.values()):
                L_soup += ((p - p_soup.to(device))**2).sum()
        
        # -----------------------------
        # 5. Total Loss & Backward
        # -----------------------------
        L_total = L_adv + lambda_geom * L_geom + lambda_soup * L_soup
        
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
