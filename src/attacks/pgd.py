import torch
import torch.nn as nn

def pgd_attack(model, x, y, epsilon, alpha, iters, device='cuda', norm='inf'):
    x_adv = x.clone().detach().to(device)
    x_orig = x.clone().detach()
    for i in range(iters):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        grad = x_adv.grad.detach()
        if norm == 'inf':
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - epsilon), x_orig + epsilon)
        elif norm == 2:
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1)
            x_adv = x_adv + alpha * grad / (grad_norm + 1e-10)
            delta = x_adv - x_orig
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1,1,1,1)
            x_adv = x_orig + delta / torch.max(delta_norm/epsilon, torch.ones_like(delta_norm))
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv
