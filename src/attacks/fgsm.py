"""
Fast Gradient Sign Method (FGSM) attack implementation.

FGSM is a single-step adversarial attack that perturbs inputs in the direction
of the gradient sign to maximize the loss.
"""

import torch
import torch.nn as nn
from typing import Union


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8/255,
    device: Union[str, torch.device] = None
) -> torch.Tensor:
    """
    Generate adversarial examples using FGSM (Fast Gradient Sign Method).
    
    Args:
        model: PyTorch model to attack
        x: Input images tensor [batch, channels, height, width]
        y: True labels tensor [batch]
        epsilon: Perturbation magnitude (L_inf norm)
        device: Device to run attack on. If None, uses same device as x.
    
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
    
    x.requires_grad = True
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial perturbation
    x_adv = x + epsilon * x.grad.sign()
    
    # Clip to valid image range
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    return x_adv.detach()
