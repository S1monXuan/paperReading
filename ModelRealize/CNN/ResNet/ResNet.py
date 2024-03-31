from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, use_1_1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1_1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)
        self.relu =  nn.ReLU(inplace = True)

    def forward(self, X):
        res_x = X
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))

        if self.conv3:
            res_x = self.conv3(res_x)
        
        X = X + res_x
        return F.relu(X)
    
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """Create a resnet block based on residual block. Resnet_block would create a block which contains several resnet.

    Args:
        input_channels (_type_): _description_
        num_channels (_type_): _description_
        num_residuals (_type_): _description_
        first_block (bool, optional): _description_. Defaults to False.
    """
    blk = []
    for i in range(num_residuals):
        # print(i, first_block)
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, strides = 2, use_1_1conv=True)
            )
        else:
            blk.append(
                Residual(num_channels, num_channels, strides = 1)
            )
    return blk

def ResNet(ratio = 1):
    """Function to create 34 Layer

    Returns:
        _type_: _description_
    """
    b1 = nn.Sequential(nn.Conv2d(1, 64 // ratio, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64 // ratio), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )
    b2 = nn.Sequential(*resnet_block(64 // ratio, 64 // ratio, num_residuals = 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64 // ratio, 128 // ratio, num_residuals = 2))
    b4 = nn.Sequential(*resnet_block(128 // ratio, 256 // ratio, num_residuals = 2))
    b5 = nn.Sequential(*resnet_block(256 // ratio, 512 // ratio, num_residuals = 2))
    
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512 // ratio, 10))

    return net
