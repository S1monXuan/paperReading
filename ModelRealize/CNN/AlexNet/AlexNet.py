import torch 
from torch import nn
import torch.optim as optim
# import Fashion-MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_AlexNet():
    """Generate AlexNet based on Papaer

    Returns:
        net: AlexNet for fashion_MNIST
    """
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )
    return net

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
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True,num_workers=8)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=True, num_workers=8)

    return train_loader, val_loader

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
            
            # Move data to gpu if using cuda
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Print current loss every 5 epochs
        # if (epoch + 1) // 5 == 0:
        #     print(f'Epoch {epoch} loss is: {loss.item()}\n')
        print(f'Epoch {epoch} loss is: {loss.item()}')

    torch.save(model, 'Self_AlexNet.pth')
    return model

def eval_model(val_loader, model, device):
    """Eval Model and get final result

    Args:
        val_loader (_type_): The val or test data
        model (_type_): trained model
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to gpu if using cuda
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}\n')


def get_device():
    """Get device, would use GPU if GPU is avaliable else use cpu

    Returns:
        _type_: _description_
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    return device

if __name__ == '__main__':
    device = get_device()
    # Create AlexNet and store it in model
    model = create_AlexNet()
    if(torch.cuda.is_available()):
        model.to(device)

    print("Loading data")
    # Create DataLoader for both Train and val data
    train_loader, val_loader = data_collection()

    # Training data

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('Training Data')
    # Train model using train_loader
    model = train_model(train_loader, model, criterion, optimizer, device, 10)

    # Load Trained Data
    print('Load Trained data')
    model = torch.load('Self_AlexNet.pth')
    if(torch.cuda.is_available()):
        print("CUDA AVAILABLE")
        model.to(device)

    print('Predict Train dataset')
    # Get res for train dataset
    eval_model(train_loader, model, device)

    print('Predict Val dataset')
    # Get res for val dataset
    eval_model(val_loader, model, device)


