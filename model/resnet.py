import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # extra: [b, in_channels, h, w] => [b, out_channels, h, w]
        self.extra = nn.Sequential()
        if in_channels != out_channels:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # short cut
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class Resnet18(nn.Module):
    def __init__(self, num_class=2) -> None:
        super(Resnet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # [b, 64, h, w] => [b, 128, h, w]
        self.resblock1 = ResBlock(16, 32, stride=3)
        # [b, 128, h, w] => [b, 256, h, w]
        self.resblock2 = ResBlock(32, 64, stride=3)
        # [b, 256, h, w] => [b, 512, h, w]
        self.resblock3 = ResBlock(64, 128, stride=2)
        # [b, 512, h, w] => [b, 512, h, w]
        self.resblock4 = ResBlock(128, 256, stride=2)
        self.fc = nn.Linear(256*3*3, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        #print('after conv:', x.shape)
        # [b, 512, h, w] => [b, 512, 1, 1]
        #x = F.adaptive_avg_pool2d(x, [1, 1])
        #print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_class=2) -> None:
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # [batch, channel, height, width]
        # [b, 64, h, w] => [b, 128, h, w]
        self.resblock1 = ResBlock(64, 128)
        # [b, 128, h, w] => [b, 256, h, w]
        self.resblock2 = ResBlock(128, 256)
        # [b, 256, h, w] => [b, 512, h, w]
        self.resblock3 = ResBlock(256, 512)
        # [b, 512, h, w] => [b, 512, h, w]
        self.resblock4 = ResBlock(512, 512)
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        # print('after conv:', x.shape)
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    #res = ResBlock(64, 128, 2)
    #tmp = torch.randn(2, 64, 32, 32)
    #out = res(tmp)
    #print('block:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18(10)
    #print('model:', model)
    out = model(x)
    print('resnet:', out.shape)
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameter size:', p)

if __name__ == '__main__':
    main()





