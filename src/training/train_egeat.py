import torch
import torch.nn.functional as F
from copy import deepcopy

# ------------------------------------------------------------
# 1. Exact Inner Maximization (Dual-Norm Adversarial Perturbation)
# ------------------------------------------------------------
def exact_delta_star(grad, eps, norm='inf'):
    """
    Compute exact first-order adversarial perturbation:
        δ* = ε * g / ||g||*
    where ||·||* is the dual norm of L_p.
    """
    if norm == 'inf':
        # Dual norm of L_inf is L1 → δ = eps * sign(g)
        return eps * grad.sign()

    g_flat = grad.view(grad.size(0), -1)

    if norm == 2:
        # dual(p=2) = p=2
        g_norm = g_flat.norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
        return eps * grad / g_norm

    if norm == 1:
        # dual(p=1) = p=inf
        g_norm = g_flat.abs().max(dim=1)[0].view(-1, 1, 1, 1) + 1e-12
        return eps * grad / g_norm

    raise ValueError("Unsupported norm type.")


# ------------------------------------------------------------
# 2. Gradient Subspace Geometric Regularizer
# ------------------------------------------------------------
def gradient_subspace_penalty(model, x, y, ensemble_models):
    """
    Computes:
        L_geom = Σ_{i<j} Tr(G_i G_j^T) / (||G_i||_F ||G_j||_F)
    using gradient vectors from model + ensemble snapshots.
    """
    if not ensemble_models or len(ensemble_models) == 0:
        return 0.0

    grads = []

    # Current model gradient
    x_req = x.clone().detach().requires_grad_(True)
    out = model(x_req)
    loss = F.cross_entropy(out, y)
    g = torch.autograd.grad(loss, x_req)[0].view(x.size(0), -1)
    grads.append(g)

    # Snapshot gradients (no state_dict loading!)
    for m in ensemble_models:
        m.eval()
        x_snap = x.clone().detach().requires_grad_(True)
        out_snap = m(x_snap)
        loss_snap = F.cross_entropy(out_snap, y)
        g_snap = torch.autograd.grad(loss_snap, x_snap)[0].view(x.size(0), -1)
        grads.append(g_snap)

    # Compute pairwise cosine/Frobenius similarities
    L = len(grads)
    penalty = 0.0
    for i in range(L):
        for j in range(i + 1, L):
            Gi = grads[i]
            Gj = grads[j]
            num = (Gi * Gj).sum(dim=1).mean()
            den = (Gi.norm(p=2, dim=1) * Gj.norm(p=2, dim=1)).mean() + 1e-12
            penalty += num / den

    return penalty / (L * (L - 1) / 2)


# ------------------------------------------------------------
# 3. Ensemble (Soup) Smoothing
# ------------------------------------------------------------
def compute_theta_soup(ensemble_snapshots):
    """Computes average of model parameters without breaking state dicts."""
    if ensemble_snapshots is None or len(ensemble_snapshots) == 0:
        return None

    # Deepcopy template
    soup = deepcopy(ensemble_snapshots[0])

    for k in soup.keys():
        for s in ensemble_snapshots[1:]:
            soup[k] += s[k]
        soup[k] /= len(ensemble_snapshots)

    return soup


# ------------------------------------------------------------
# 4. Main EGEAT Epoch
# ------------------------------------------------------------
def train_egeat_epoch(
    model, dataloader, optimizer, device='cuda',
    lambda_geom=0.1, lambda_soup=0.05,
    epsilon=8/255, p_norm='inf',
    ensemble_snapshots=None
):
    """
    One epoch of EXACT EGEAT training.
    """
    model.train()

    # Precompute soup centroid
    theta_soup = compute_theta_soup(ensemble_snapshots)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)

        # ==========================================================
        # 1. Exact Inner Maximization (Dual Norm)
        # ==========================================================
        logits = model(x)
        loss_clean = F.cross_entropy(logits, y)
        grad_x = torch.autograd.grad(loss_clean, x, create_graph=False)[0]
        delta_star = exact_delta_star(grad_x, epsilon, norm=p_norm)
        x_adv = torch.clamp(x + delta_star, 0.0, 1.0)

        # ==========================================================
        # 2. Adversarial Loss
        # ==========================================================
        logits_adv = model(x_adv)
        L_adv = F.cross_entropy(logits_adv, y)

        # ==========================================================
        # 3. Geometric Regularizer (Correct Formula)
        # ==========================================================
        L_geom = gradient_subspace_penalty(model, x.detach(), y, ensemble_snapshots)

        # ==========================================================
        # 4. Ensemble Smoothing Loss
        # ==========================================================
        L_soup = 0.0
        if theta_soup is not None:
            for p, key in zip(model.parameters(), theta_soup.keys()):
                p_s = theta_soup[key].to(device)
                L_soup += ((p - p_s) ** 2).sum()

        # ==========================================================
        # 5. Final EGEAT Loss
        # ==========================================================
        L_total = (
            L_adv
            + lambda_geom * L_geom
            + lambda_soup * L_soup
        )

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
