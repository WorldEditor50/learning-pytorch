import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):

    def __init__(self) -> None:
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self.criteon = nn.CrossEntropyLoss()


    def forward(self, x):
        batchsize = x.size(0)
        # [batchsize, 3, 32, 32] => [batchsize, 16, 5, 5]
        x = self.conv_unit(x)
        # flatten
        x = x.view(batchsize, 16*5*5)
        # [batchsize, 16*5*5] => [batchsize, 10]
        logits = self.fc_unit(x)
        return logits
def main():
    tmp = torch.randn(2, 3, 32, 32)
    lenet5 = Lenet5()
    out = lenet5(tmp)
    print('lenet5 out:', out.shape)
    return

if __name__ == '__main__':
    main()
