import torch
import torch.nn as nn

def cw_l2_attack(model, x, y, c=1e-2, iters=100, lr=0.01, device='cuda'):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    optimizer = torch.optim.Adam([x_adv], lr=lr)
    for i in range(iters):
        optimizer.zero_grad()
        output = model(x_adv)
        f_loss = torch.clamp(output.gather(1, y.unsqueeze(1)) - output.max(1)[0].unsqueeze(1), min=0)
        loss = (x_adv - x).pow(2).sum() + c * f_loss.sum()
        loss.backward()
        optimizer.step()
        x_adv.data = torch.clamp(x_adv.data, 0, 1)
    return x_adv.detach()
