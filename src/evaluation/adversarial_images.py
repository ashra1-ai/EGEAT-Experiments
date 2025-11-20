"""
Adversarial example generation and visualization utilities.
"""

import torch
import numpy as np
import imageio
import os
from typing import List, Tuple, Union
from src.attacks.pgd import pgd_attack


def generate_adv_examples(
    src_model: torch.nn.Module,
    tgt_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = None,
    n_samples: int = 24,
    epsilon: float = 8/255,
    alpha: float = 2/255,
    iters: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate adversarial examples using source model and test on target model.
    
    Args:
        src_model: Source model to generate adversarial examples
        tgt_model: Target model to test adversarial examples on
        dataloader: DataLoader for input images
        device: Device to run computation on. If None, uses same device as models.
        n_samples: Number of adversarial examples to generate
        epsilon: Perturbation magnitude for PGD attack
        alpha: Step size for PGD attack
        iters: Number of PGD iterations
    
    Returns:
        List of tuples (original_image, adversarial_image) as numpy arrays
    """
    if device is None:
        device = next(src_model.parameters()).device
    else:
        device = torch.device(device)
    
    src_model.eval()
    tgt_model.eval()
    imgs = []
    cnt = 0
    
    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Generate adversarial examples
        X_adv_src = pgd_attack(
            src_model, X, y,
            epsilon=epsilon,
            alpha=alpha,
            iters=iters,
            device=device
        )
        
        # Test on target model (optional, for verification)
        with torch.no_grad():
            _ = tgt_model(X_adv_src)
        
        # Convert to numpy
        X_np = X.cpu().numpy()
        X_adv_np = X_adv_src.cpu().numpy()
        
        for i in range(X_np.shape[0]):
            imgs.append((X_np[i], X_adv_np[i]))
            cnt += 1
            if cnt >= n_samples:
                break
        if cnt >= n_samples:
            break
    
    return imgs

def save_adv_images(
    img_tuples: List[Tuple[np.ndarray, np.ndarray]],
    save_path: str
):
    """
    Save adversarial example images to disk.
    
    Args:
        img_tuples: List of (original, adversarial) image tuples
        save_path: Directory to save images
    """
    os.makedirs(save_path, exist_ok=True)
    
    for idx, (orig, adv) in enumerate(img_tuples):
        # Handle different image formats
        if len(orig.shape) == 3 and orig.shape[0] in [1, 3]:
            # CHW format
            orig_img = orig.transpose(1, 2, 0)
            adv_img = adv.transpose(1, 2, 0)
        else:
            orig_img = orig
            adv_img = adv
        
        # Normalize to [0, 255]
        if orig_img.max() <= 1.0:
            orig_img = (orig_img * 255).astype(np.uint8)
            adv_img = (adv_img * 255).astype(np.uint8)
        else:
            orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
            adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        
        # Handle grayscale
        if len(orig_img.shape) == 2:
            orig_img = np.stack([orig_img] * 3, axis=-1)
            adv_img = np.stack([adv_img] * 3, axis=-1)
        
        imageio.imwrite(os.path.join(save_path, f"orig_{idx:04d}.png"), orig_img)
        imageio.imwrite(os.path.join(save_path, f"adv_{idx:04d}.png"), adv_img)


def evaluate_model_on_adv(
    tgt_model: torch.nn.Module,
    img_tuples: List[Tuple[np.ndarray, np.ndarray]],
    labels: List[int],
    device: Union[str, torch.device] = None
) -> float:
    """
    Evaluate model accuracy on adversarial examples.
    
    Args:
        tgt_model: Model to evaluate
        img_tuples: List of (original, adversarial) image tuples
        labels: True labels for the images
        device: Device to run evaluation on. If None, uses same device as model.
    
    Returns:
        float: Accuracy on adversarial examples
    """
    if device is None:
        device = next(tgt_model.parameters()).device
    else:
        device = torch.device(device)
    
    tgt_model.eval()
    correct = 0
    total = 0
    
    for (orig, adv), label in zip(img_tuples, labels):
        # Convert to tensor
        if len(adv.shape) == 3 and adv.shape[0] in [1, 3]:
            adv_tensor = torch.tensor(adv, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            adv_tensor = torch.tensor(adv, dtype=torch.float32).to(device)
            if len(adv_tensor.shape) == 3:
                adv_tensor = adv_tensor.unsqueeze(0)
        
        with torch.no_grad():
            preds = tgt_model(adv_tensor).argmax(dim=1)
        
        if preds.item() == label:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0.0
    return acc