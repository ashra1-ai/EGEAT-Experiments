import torch
import numpy as np
import imageio
from src.attacks.pgd import pgd_attack

def generate_adv_examples(src_model, tgt_model, dataloader, device='cpu', n_samples=24):
    src_model.eval(); tgt_model.eval()
    imgs = []
    cnt = 0
    for X,y in dataloader:
        X = X.to(device); y = y.to(device)
        X_adv_src = pgd_attack(src_model, X, y, eps=8/255, alpha=2/255, steps=10)
        with torch.no_grad():
            _ = tgt_model(X_adv_src)
        X = X.cpu().numpy(); X_adv = X_adv_src.cpu().numpy()
        for i in range(X.shape[0]):
            imgs.append((X[i], X_adv[i]))
            cnt +=1
            if cnt>=n_samples: break
        if cnt>=n_samples: break
    return imgs

def save_adv_images(img_tuples, save_path):
    for idx, (orig, adv) in enumerate(img_tuples):
        orig_img = ((orig.transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)
        adv_img = ((adv.transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)
        imageio.imwrite(f"{save_path}/orig_{idx}.png", orig_img)
        imageio.imwrite(f"{save_path}/adv_{idx}.png", adv_img)
def evaluate_model_on_adv(tgt_model, img_tuples, device='cpu'):
    tgt_model.eval()
    correct = 0
    total = 0
    for orig, adv in img_tuples:
        adv_tensor = torch.tensor(adv).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = tgt_model(adv_tensor).argmax(dim=1)
        total += 1
        if preds.item() == 1:  # Assuming label '1' is the true label for simplicity
            correct += 1
    acc = correct / total
    return acc