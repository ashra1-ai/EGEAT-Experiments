import torch
from src.training.train_egeat import train_egeat_epoch

def quick_ablation_run(lambda_1, lambda_2, model, train_loader, test_loader, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_egeat_epoch(model, train_loader, optimizer, device=device, lambda_geom=lambda_1, lambda_soup=lambda_2)
    model.eval()
    correct, total = 0, 0
    for i,(X,y) in enumerate(test_loader):
        if i>5: break
        with torch.no_grad():
            preds = model(X.to(device)).argmax(dim=1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
    acc = correct / total
    return acc
