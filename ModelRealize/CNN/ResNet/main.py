from model import get_device, save_model, train_model, eval_model, load_model, cuda_available
from ResNet import ResNet
from data_set import get_fashionMNIST_data

from torch import nn
import torch.optim as optim

if __name__ == '__main__':
    device = get_device()
    ratio = 4
    model_path = 'Self_ResNet_34.pth'

    print('Loading data...')
    train_loader, val_loader = get_fashionMNIST_data('../Data/FashionMNIST/')

    print('Generating Model...')
    # Generate Model
    model = ResNet(ratio)
    if cuda_available():
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print('Training Model...')
    model = train_model(train_loader, model, criterion, optimizer, device, 10)
    print('Saving Model...')
    save_model(model, model_path)

    # print('Loading model...')
    # model = load_model(model_path)
    # if cuda_available():
    #     print('turn model to cuda')
    #     model.to(device)

    print('Predict Train dataset')
    # Get res for train dataset
    eval_model(train_loader, model, device)

    print('Predict Val dataset')
    # Get res for val dataset
    eval_model(val_loader, model, device)