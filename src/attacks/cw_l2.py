"""
Carlini & Wagner L2 attack implementation.

The C&W attack is an optimization-based attack that minimizes a combination of
L2 perturbation magnitude and a margin-based loss function.
"""

import torch
import torch.nn as nn
from typing import Union


def cw_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1e-2,
    iters: int = 100,
    lr: float = 0.01,
    device: Union[str, torch.device] = None,
    targeted: bool = False
) -> torch.Tensor:
    """
    Generate adversarial examples using C&W L2 attack.
    
    The attack minimizes: ||x_adv - x||_2^2 + c * f(x_adv, y)
    where f is a margin-based loss function.
    
    Args:
        model: PyTorch model to attack
        x: Input images tensor [batch, channels, height, width]
        y: True labels tensor [batch]
        c: Weight for the margin loss term
        iters: Number of optimization iterations
        lr: Learning rate for optimizer
        device: Device to run attack on. If None, uses same device as x.
        targeted: If True, performs targeted attack
    
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
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    
    # Optimizer for adversarial example
    optimizer = torch.optim.Adam([x_adv], lr=lr)
    
    # C&W attack iterations
    for i in range(iters):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x_adv)
        
        # Margin-based loss: f(x) = max(0, max_{i≠t}(Z(x)_i) - Z(x)_t)
        # where Z(x) are logits and t is the target class
        if targeted:
            # For targeted: minimize Z(x)_t - max_{i≠t}(Z(x)_i)
            target_logits = output.gather(1, y.unsqueeze(1))
            other_logits = output.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other_logits = other_logits.max(1)[0].unsqueeze(1)
            f_loss = torch.clamp(max_other_logits - target_logits, min=0.0)
        else:
            # For untargeted: maximize max_{i≠y}(Z(x)_i) - Z(x)_y
            correct_logits = output.gather(1, y.unsqueeze(1))
            other_logits = output.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float('-inf'))
            max_other_logits = other_logits.max(1)[0].unsqueeze(1)
            f_loss = torch.clamp(correct_logits - max_other_logits, min=0.0)
        
        # Total loss: L2 distance + margin loss
        l2_dist = (x_adv - x_orig).pow(2).sum(dim=(1, 2, 3)).mean()
        margin_loss = f_loss.sum()
        loss = l2_dist + c * margin_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Clip to valid image range
        x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
    
    return x_adv.detach()
