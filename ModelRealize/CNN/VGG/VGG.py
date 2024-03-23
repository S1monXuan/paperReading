from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    """A VGG BLOCK, each conv layer is followed with a ReLU layer.

    Args:
        num_convs (_type_): number of VGG block
        in_channels (_type_): input_channel number 
        out_channels (_type_): output_channel number

    Returns:
        _type_: return a sequential layer for this VGG block
    """
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def VGG_11(in_channels, conv_arch):
    vgg_blocks = []
    in_channels = in_channels
    for (num_convs, out_channels) in conv_arch:
        vgg_blocks.append(
            vgg_block(num_convs, in_channels, out_channels)
        )
        in_channels = out_channels
    
    VGG = nn.Sequential(*vgg_blocks, nn.Flatten(),
                        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                        nn.Linear(4096, 4096), nn.ReLU(),
                        nn.Linear(4096, 10)                        
                        )
    return VGG
