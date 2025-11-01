import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_gradient_similarity_matrix(S, title="Gradient Cosine Similarity", save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(S, annot=True, cmap='viridis')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Model')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_loss_landscape(loss_values, title="Loss Landscape", save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(loss_values)
    plt.title(title)
    plt.xlabel("Interpolation Step")
    plt.ylabel("Loss")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_adversarial_examples_grid(x_adv, n=8, save_path=None):
    plt.figure(figsize=(10,10))
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        img = x_adv[i].cpu().detach().numpy()
        if img.shape[0]==1:  # grayscale
            plt.imshow(img[0], cmap='gray')
        else:
            plt.imshow(np.transpose(img, (1,2,0)))
        plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
def plot_accuracy_bars(accuracies, model_names, title="Model Accuracies", save_path=None):
    plt.figure(figsize=(8,6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()