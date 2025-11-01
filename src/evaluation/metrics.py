import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(model, data_loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def gradient_cosine_similarity(model_list, data_loader, device='cuda'):
    similarities = []
    criterion = nn.CrossEntropyLoss()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        grads = []
        for model in model_list:
            model.zero_grad()
            x.requires_grad = True
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            grads.append(x.grad.detach().flatten(1))
        for i in range(len(grads)):
            for j in range(i+1, len(grads)):
                cos_sim = F.cosine_similarity(grads[i], grads[j], dim=1).mean().item()
                similarities.append(cos_sim)
    return sum(similarities)/len(similarities)

def ece(model, data_loader, n_bins=15, device='cuda'):
    model.eval()
    confidences, predictions, labels = [], [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(1)
            confidences.append(conf.cpu())
            predictions.append(pred.cpu())
            labels.append(y.cpu())
    confidences = torch.cat(confidences); predictions = torch.cat(predictions); labels = torch.cat(labels)
    bins = torch.linspace(0, 1, n_bins+1)
    ece_score = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).float().mean()
            conf = confidences[mask].mean()
            ece_score += (mask.float().mean()) * abs(acc - conf)
    return ece_score.item()
def nll_loss(model, data_loader, device='cuda'):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    return total_loss / total_samples