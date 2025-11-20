"""
Evaluation metrics for model performance assessment.

This module provides functions for computing accuracy, calibration error,
and other evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


def accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on. If None, uses same device as model.
    
    Returns:
        float: Accuracy score (0-1)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def gradient_cosine_similarity(
    model_list: list,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None
) -> float:
    """
    Compute average cosine similarity between gradients of different models.
    
    Args:
        model_list: List of PyTorch models
        data_loader: DataLoader for computing gradients
        device: Device to run computation on. If None, uses same device as first model.
    
    Returns:
        float: Average cosine similarity between model gradients
    """
    if device is None:
        device = next(model_list[0].parameters()).device
    else:
        device = torch.device(device)
    
    similarities = []
    criterion = nn.CrossEntropyLoss()
    
    for x, y in data_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        grads = []
        
        for model in model_list:
            model.zero_grad()
            x_clone = x.clone().requires_grad_(True)
            output = model(x_clone)
            loss = criterion(output, y)
            loss.backward()
            grads.append(x_clone.grad.detach().flatten(1))
        
        for i in range(len(grads)):
            for j in range(i+1, len(grads)):
                cos_sim = F.cosine_similarity(grads[i], grads[j], dim=1).mean().item()
                similarities.append(cos_sim)
    
    return sum(similarities) / len(similarities) if similarities else 0.0

def ece(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    n_bins: int = 15,
    device: Union[str, torch.device] = None
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        n_bins: Number of bins for calibration curve
        device: Device to run evaluation on. If None, uses same device as model.
    
    Returns:
        float: ECE score
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    confidences, predictions, labels = [], [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(1)
            confidences.append(conf.cpu())
            predictions.append(pred.cpu())
            labels.append(y.cpu())
    
    confidences = torch.cat(confidences)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    
    bins = torch.linspace(0, 1, n_bins + 1)
    ece_score = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).float().mean()
            conf = confidences[mask].mean()
            ece_score += (mask.float().mean()) * abs(acc - conf)
    
    return ece_score.item()


def nll_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None
) -> float:
    """
    Compute Negative Log-Likelihood (NLL) loss.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on. If None, uses same device as model.
    
    Returns:
        float: Average NLL loss
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    return total_loss / total_samples if total_samples > 0 else float('inf')