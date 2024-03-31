import torch 
import datetime

def cuda_available():
    """Check if Cuda is available

    Returns:
        boolean: True if cuda is avaliable, else False
    """   
    return torch.cuda.is_available()

def get_device():
    """Get device, would use GPU if GPU is avaliable else use cpu

    Returns:
        _type_: _description_
    """
    device = torch.device("cuda" if cuda_available() else "cpu")
    print(f'Using {device}')
    return device

def save_model(model, model_path):
    """Save model locally

    Args:
        model (_type_): _description_
        model_path (_type_): _description_
    """
    torch.save(model, model_path)

def train_model(train_loader, model, criterion, optimizer, device, num_epochs = 10):
    """Train AlexNet for further predition

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
    """
    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch} start training: at time: {datetime.datetime.now()}')
        for images, labels in train_loader:
            if cuda_available():
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss is: {loss.item()}')
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
            if cuda_available():
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            total += outputs.size(0)
            correct += (predicts == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}\n')
    
def load_model(model_path):
    return torch.load(model_path)