import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=128, val_split=0.1, device='cuda'):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_cifar10_loaders(batch_size=128, val_split=0.1, device='cuda'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Optional: DREBIN loader placeholder
def get_drebin_loaders(batch_size=128, val_split=0.1):
    # Load from CSV / Numpy arrays (binary features)
    import numpy as np
    from torch.utils.data import TensorDataset

    X = np.load('./data/DREBIN/X.npy')
    y = np.load('./data/DREBIN/y.npy')
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = None  # Load separately if needed
    return train_loader, val_loader, test_loader
