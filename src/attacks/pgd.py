"""
Projected Gradient Descent (PGD) attack implementation.

PGD is a multi-step iterative attack that performs gradient ascent in the
adversarial direction, projecting back to the epsilon-ball at each step.
"""

import torch
import torch.nn as nn
from typing import Union, Literal


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8/255,
    alpha: float = 2/255,
    iters: int = 10,
    device: Union[str, torch.device] = None,
    norm: Literal['inf', 2] = 'inf',
    random_start: bool = True,
    targeted: bool = False
) -> torch.Tensor:
    """
    Generate adversarial examples using PGD (Projected Gradient Descent).
    
    Args:
        model: PyTorch model to attack
        x: Input images tensor [batch, channels, height, width]
        y: True labels tensor [batch]
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        iters: Number of PGD iterations
        device: Device to run attack on. If None, uses same device as x.
        norm: Norm to use for perturbation ('inf' for L_inf, 2 for L2)
        random_start: Whether to start from random perturbation
        targeted: If True, performs targeted attack (minimize loss for target class)
    
    Returns:
        torch.Tensor: Adversarial examples with same shape as x
    """
    if device is None:
        device = x.device
    else:
        device = torch.device(device)
    
    model.eval()
    x = x.clone().detach().to(device)
    y = y.to(device)
    x_orig = x.clone()
    
    # Initialize adversarial example
    if random_start:
        if norm == 'inf':
            x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        else:  # L2 norm
            delta = torch.randn_like(x)
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
            delta = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-10) * epsilon
            x_adv = x + delta
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x.clone()
    
    # PGD iterations
    for i in range(iters):
        x_adv.requires_grad = True
        
        # Forward pass
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        
        if targeted:
            loss = -loss  # Minimize loss for targeted attack
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        
        # Update adversarial example
        if norm == 'inf':
            # L_inf: sign of gradient
            x_adv = x_adv + alpha * grad.sign()
            # Project back to epsilon-ball
            x_adv = torch.min(torch.max(x_adv, x_orig - epsilon), x_orig + epsilon)
        else:  # L2 norm
            # L2: normalized gradient
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-10)
            x_adv = x_adv + alpha * grad_normalized
            # Project back to epsilon-ball
            delta = x_adv - x_orig
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True)
            delta_normalized = delta / (delta_norm.view(-1, 1, 1, 1) + 1e-10)
            x_adv = x_orig + delta_normalized * torch.minimum(
                delta_norm.view(-1, 1, 1, 1) / epsilon,
                torch.ones_like(delta_norm.view(-1, 1, 1, 1))
            ) * epsilon
        
        # Clip to valid image range
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach()
    
    return x_adv
