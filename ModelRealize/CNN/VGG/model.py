import torch 
from torch import nn
import torch.optim as optim
# import Fashion-MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from data_set import data_collection
from VGG import VGG_11

def train_model(train_loader, model, criterion, optimizer, device, num_epochs = 10):
    """Train AlexNet for further predition

    Args:
        train_loader (_type_): train_dataloader
        model (_type_): AlexNet model that will be trained
        criterion (_type_): criterion
        optimizer (_type_): select optimizer
        num_epochs (int, optional): Epochs time. Defaults to 10.

    Returns:
        _type_: _description_
    """
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            if cuda_available:
                images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss is: {loss.item()}')
    
    torch.save(model, 'Self_VGG_11.pth')
    return model


def eval_model(val_loader, model, device):
    """Eval Model and get final result

    Args:
        val_loader (_type_): The val or test data
        model (_type_): trained model
    """
    model.eval()
    correct, total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            if cuda_available():
                images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            total += outputs.size(0)
            correct += (predicts == outputs).sum().item()

    print(f'Accuracy: {100 * correct / total}\n')


def get_device():
    """Get device, would use GPU if GPU is avaliable else use cpu

    Returns:
        _type_: _description_
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    return device

def cuda_available():
    """Check if Cuda is available

    Returns:
        boolean: True if cuda is avaliable, else False
    """
    return torch.cuda.is_available()

if __name__ == '__main__':
    device = get_device()
    model = VGG_11()

    if cuda_available():
        model.to(device)

    print("Loading data")
    # Create DataLoader for both Train and val data
    train_loader, val_loader = data_collection()

    # Tranining data
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Training Data')
    # Train model using train_loader
    model = train_model(train_loader, model, criterion, optimizer, device, 10)

    print('Predict Train dataset')
    # Get res for train dataset
    eval_model(train_loader, model, device)

    print('Predict Val dataset')
    # Get res for val dataset
    eval_model(val_loader, model, device)