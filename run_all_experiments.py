import os
import torch
from torch.utils.data import DataLoader
from src.models import SimpleCNN
from src.training.train_egeat import train_egeat_epoch
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.cw_l2 import cw_l2_attack
from src.evaluation.metrics import accuracy, ece
from src.evaluation.gradient_similarity import gradient_subspace_similarity
from src.evaluation.ensemble_variance import ensemble_variance
from src.evaluation.loss_landscape import scan_1d_loss, scan_2d_loss
from src.evaluation.transferability import transferability_matrix
from src.evaluation.adversarial_images import generate_adv_examples, save_adv_images
from src.utils.data_loader import dataloaders
import src.utils.viz_utils as viz

# -------------------
# Config
# -------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LR = 2e-4
EPOCHS = 5
LAMBDA_GEOM = 0.1
LAMBDA_SOUP = 0.05
ENSMBLE_SIZE = 5
EPSILON = 8/255
ALPHA = 2/255
PGD_ITERS = 10
CW_ITERS = 50
SAVE_DIR = "results"

# Choose dataset: 'cifar10', 'mnist', 'drebin'
DATASET = 'cifar10'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/adv_images", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/figures", exist_ok=True)

# -------------------
# Load dataset
# -------------------
if DATASET.lower() == 'cifar10':
    train_loader, val_loader, test_loader = dataloaders.get_cifar10_loaders(batch_size=BATCH_SIZE)
elif DATASET.lower() == 'mnist':
    train_loader, val_loader, test_loader = dataloaders.get_mnist_loaders(batch_size=BATCH_SIZE)
elif DATASET.lower() == 'drebin':
    train_loader, val_loader, test_loader = dataloaders.get_drebin_loaders(batch_size=BATCH_SIZE)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}")

# -------------------
# Initialize ensemble
# -------------------
models = [SimpleCNN().to(DEVICE) for _ in range(ENSMBLE_SIZE)]

# -------------------
# Train EGEAT models
# -------------------
for i, model in enumerate(models):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Training model {i+1}/{ENSMBLE_SIZE}")
    for epoch in range(EPOCHS):
        train_egeat_epoch(model, train_loader, optimizer, device=DEVICE,
                          lambda_geom=LAMBDA_GEOM, lambda_soup=LAMBDA_SOUP)
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_{i}.pt")

# -------------------
# Compute Metrics
# -------------------
for i, model in enumerate(models):
    acc = accuracy(model, test_loader, device=DEVICE)
    ece_score = ece(model, test_loader, device=DEVICE)
    print(f"Model {i}: Acc={acc:.4f}, ECE={ece_score:.4f}")

# Gradient subspace similarity
sim_matrix = gradient_subspace_similarity(models, test_loader, device=DEVICE)
viz.save_heatmap(sim_matrix, f"{SAVE_DIR}/figures/grad_similarity.png", title="Gradient Subspace Similarity")

# Ensemble variance
Kvals, vars_ = ensemble_variance(models, test_loader, device=DEVICE)
viz.save_line(Kvals, vars_, f"{SAVE_DIR}/figures/ensemble_variance.png", xlabel="K", ylabel="Variance", title="Ensemble Variance")

# Transferability
P = transferability_matrix(models, test_loader, device=DEVICE)
viz.save_heatmap(P, f"{SAVE_DIR}/figures/transferability.png", title="Transferability Matrix")

# -------------------
# Adversarial Examples
# -------------------
adv_samples = generate_adv_examples(models[0], models[1], test_loader, device=DEVICE, n_samples=16)
save_adv_images(adv_samples, f"{SAVE_DIR}/adv_images")
viz.save_image_grid(adv_samples, f"{SAVE_DIR}/figures/adv_examples.png", nrow=4, title="Adversarial Examples")

# -------------------
# Loss Landscapes
# -------------------
alphas1D, losses1D = scan_1d_loss(models[0], test_loader, device=DEVICE)
viz.save_line(alphas1D, losses1D, f"{SAVE_DIR}/figures/loss_1d.png", xlabel="Alpha", ylabel="Loss", title="1D Loss Landscape")

alphas2D, betas2D, losses2D = scan_2d_loss(models[0], test_loader, device=DEVICE)
viz.save_contour(alphas2D, betas2D, losses2D, f"{SAVE_DIR}/figures/loss_2d.png", title="2D Loss Landscape")

print("All experiments completed. Results saved in:", SAVE_DIR)
