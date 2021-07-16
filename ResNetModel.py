import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cov3 = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn1(self.conv1(x)))

        identity = x
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

  # 50 layers ResNet so blocks[1, 1, 3, 4, 6, 3]
class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels=3, num_classes=1000 ):  #[3,4,6,3]
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, self.in_channels, kernel_size=(7, 7), stride=(3, 3), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    # layers
    #self.layer1 = .....
    #self.layer2 = ......
        # ResNet Layers
        self.layer1 = self.create_layers(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.create_layers(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.create_layers(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.create_layers(block, layers[3], out_channels=524, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def create_layers(self,block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                              nn.BatchNorm2d(out_channels*4))
            )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

        self.in_channels = out_channels*4

        for i in range(num_residual_block - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def Resnet50(img_channels=3, num_class=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_class)

def test():
    resnet = Resnet50()
    y = resnet(torch.randn(4, 3, 224, 224))
    print(y.shape)

test()
