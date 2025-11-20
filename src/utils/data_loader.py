import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import numpy as np


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# MNIST Loader
# ============================================================
def get_mnist_loaders(batch_size=128, val_split=0.1, seed=123):
    """
    Returns MNIST train/val/test loaders with reproducible splits.
    """
    seed_everything(seed)

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


# ============================================================
# CIFAR-10 Loader
# ============================================================
def get_cifar10_loaders(batch_size=128, val_split=0.1, seed=123):
    """
    CIFAR-10 loaders with augmentations aligned with EGEAT paper.
    """
    seed_everything(seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform_train)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform_test)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


# ============================================================
# DREBIN Loader
# ============================================================
def get_drebin_loaders(batch_size=128, val_split=0.1, seed=123):
    """
    Loads DREBIN binary feature vectors from Numpy arrays.
    """
    seed_everything(seed)

    X = np.load('./data/DREBIN/X.npy')
    y = np.load('./data/DREBIN/y.npy')

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        None  # test set loaded separately
    )