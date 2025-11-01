import torch
import torch.nn as nn

def fgsm_attack(model, x, y, epsilon, device='cuda'):
    x = x.clone().detach().to(device)
    x.requires_grad = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv
