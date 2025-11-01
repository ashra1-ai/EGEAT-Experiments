import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_heatmap(mat, path, title="", cmap="viridis"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_contour(alphas, betas, losses, path, title="Loss surface"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.contourf(betas, alphas, losses, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_image_grid(imgs, path, nrow=8, normalize=True, title=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = len(imgs)
    cols = nrow
    rows = (N + cols -1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < N:
            im = imgs[i]
            if normalize:
                im = (im+1)/2  # assume [-1,1] range
            axes[i].imshow(np.clip(im,0,1))
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_line(xs, ys, path, xlabel="", ylabel="", title=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
if __name__ == "__main__":
    # Example usage of save_heatmap
    mat = np.random.rand(10,10)
    save_heatmap(mat, "output/heatmap.png", title="Random Heatmap")

    # Example usage of save_contour
    alphas = np.linspace(-3,3,100)
    betas = np.linspace(-3,3,100)
    A, B = np.meshgrid(betas, alphas)
    losses = np.sin(np.sqrt(A**2 + B**2))
    save_contour(alphas, betas, losses, "output/contour.png", title="Sine Contour")

    # Example usage of save_image_grid
    imgs = [np.random.rand(32,32,3) for _ in range(20)]
    save_image_grid(imgs, "output/image_grid.png", nrow=5, title="Random Images")

    # Example usage of save_line
    xs = list(range(10))
    ys = [x**2 for x in xs]
    save_line(xs, ys, "output/line_plot.png", xlabel="X", ylabel="Y", title="X vs Y^2")