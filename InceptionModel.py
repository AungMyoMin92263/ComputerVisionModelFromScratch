import torch
import torch.nn as nn


class GoogleNetModel(nn.Module):
    def __init__(self, in_channels=3, num_class = 1000):
        super(GoogleNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # In this order in_channels out1x1, red3x3, out3x3, red5x5, out5x5, maxpool3x3
        self.inception3a = InceptionBlcok(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlcok(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception4a = InceptionBlcok(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlcok(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlcok(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlcok(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlcok(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.inception5a = InceptionBlcok(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlcok(832, 384, 192, 384, 48, 128, 128)

        self.averagepool= nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.averagepool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionBlcok(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, maxpool3x3):
        super(InceptionBlcok, self).__init__()

        self.branch1 = ConvBlock(in_channels, out1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out1x1, kernel_size=1),
            ConvBlock(out1x1, out3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out1x1, kernel_size=1),
            ConvBlock(out1x1, out5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, maxpool3x3, kernel_size=1)
        )

    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): ##keywords arguments would be like kernel_size=(1,1), or (2,2), padding=(1,1)
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchNorm(self.conv(x)))



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GoogleNetModel()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)