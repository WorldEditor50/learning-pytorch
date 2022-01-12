import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.functional import Tensor

class LinearRegression(torch.nn.Module):

    def __init__(self) -> None:
        super(LinearRegression, self).__init__()
        self.lr = 0.01
        self.epoches = 10000
        self.model = torch.nn.Linear(in_features=1, out_features=1)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),lr=self.lr)
        self.floss = torch.nn.MSELoss()
        return

    def forward(self, x):
        out = self.model(x)
        return out
    
    def train(self, data):
        x = data["x"]
        y = data["y"]
        for epoch in range(0, self.epoches):
            yp = self.model(x)
            loss = self.floss(yp, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 500 == 0:
                print("epoch:{}, loss = {}".format(epoch, loss.item()))
        torch.save(self.model.state_dict, "linear.pth")
        return

def generateData():
    x = torch.linspace(start=0, end=1, steps=100)
    x = torch.unsqueeze(x, dim=1)
    y = 2*x + torch.rand(x.size())

    plt.scatter(x.numpy(), y.numpy(), c=x.numpy())
    plt.show()
    data = {"x":x, "y":y}
    return data

if  __name__ == '__main__':
    data = generateData()
    linear = LinearRegression()
    linear.train(data)
    x =Tensor(1)
    x[0] = 1
    y = linear.forward(x)
    print("predict:{}".format(y))
    print("done!")
