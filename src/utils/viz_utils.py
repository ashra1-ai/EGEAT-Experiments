import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ============================================================
# Heatmap
# ============================================================
def save_heatmap(mat, path, title="", cmap="viridis"):
    ensure_dir(path)
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Loss Landscape Contour
# ============================================================
def save_contour(alphas, betas, losses, path, title="Loss surface"):
    ensure_dir(path)
    plt.figure(figsize=(6,5))
    plt.contourf(betas, alphas, losses, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Grid of Images (Adversarial Examples)
# ============================================================
def save_image_grid(imgs, path, nrow=8, normalize=True, title=None):
    ensure_dir(path)
    N = len(imgs)
    cols = nrow
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis('off')
        if i < N:
            img = imgs[i]
            if normalize:  # assume [-1,1]
                img = (img + 1) / 2
            axes[i].imshow(np.clip(img, 0, 1))

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Line Plot (Accuracy curves, Loss curves, etc.)
# ============================================================
def save_line(xs, ys, path, xlabel="", ylabel="", title=""):
    ensure_dir(path)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# Testing with dummy plots
if __name__ == "__main__":
    save_heatmap(np.random.rand(10,10), "output/heatmap.png", title="Random Heatmap")
    save_contour(np.linspace(-3,3,50), np.linspace(-3,3,50),
                 np.random.rand(50,50),
                 "output/contour.png", title="Random Contour")
    save_image_grid([np.random.rand(32,32,3) for _ in range(16)], "output/grid.png")
    save_line(list(range(10)), [x**2 for x in range(10)], "output/line.png",
              xlabel="x", ylabel="x²", title="Quadratic")