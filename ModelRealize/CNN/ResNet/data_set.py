from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def transform():
    """
    output a transform which would 
        1. reshape input data to size(224, 224),
        2. Transform the datatype to tensor
        # 3. normalize the data to 0.5 0.5
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform

def get_fashionMNIST_data(data_folder):
    """ Get FashionMNIST data and turn it to dataloader type

    Args:
        data_folder (_type_): url for data_folder 
        transform (_type_): transform methods

    Returns:
        train_dataloader and test_dataloader
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # get train and test data
    train_datasets = datasets.FashionMNIST(data_folder, train=True, download=True, transform=transform)
    test_datasets = datasets.FashionMNIST(data_folder, train=False, download=True, transform=transform)

    # get DataLoader based on datasets
    train_dataloader = DataLoader(train_datasets, batch_size=256, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_datasets, batch_size=256, shuffle=True, num_workers=8)

    return train_dataloader, test_dataloader