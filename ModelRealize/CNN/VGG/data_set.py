import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_collection():
    """Download FashionMNIST into '../Data/' folder and transform it into size 224 * 224 for further use.

    Returns:
        train_loader:   DataLoader for train data, with batchsize 64
        val_loader:     DataLoader for val data, with batchsize 64
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    # Load Fashion-MNIST dataset
    train_datasets = datasets.FashionMNIST('../Data/FashionMNIST/', train=True, download=True, transform=transform)
    val_datasets = datasets.FashionMNIST('../Data/FashionMNIST/', train=False, download=True, transform=transform)

    # Data Loaders
    train_loader = DataLoader(train_datasets, batch_size=128, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_datasets, batch_size=128, shuffle=True, num_workers=4)

    return train_loader, val_loader