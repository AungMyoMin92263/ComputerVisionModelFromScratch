import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

VGG_types = {
            'VGG12': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512,'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,512,'M', 512, 512,512,512, 'M']
}

# 4096  and 4096 fully connected layers

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_class=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return  x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for channels in architecture:

            if type(channels) == int:
                out_channels = channels
                layers += [nn.Conv2d(in_channels, out_channels,
                                     kernel_size = (3, 3), stride=(1,1), padding=(1, 1)),
                                     ]
                in_channels = channels
            elif channels == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        return nn.Sequential(*layers)   ## * is for unpacking the layers like conv2d and batchnorm and as there are no batchnorm, don't need to



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG_net(3).to(device)
    x = torch.randn(15, 3, 224, 224).to(device)
    print(model.conv_layers[0])
    print(model(x).shape)
