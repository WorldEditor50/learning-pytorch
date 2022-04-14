import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
#from model.lenet5 import Lenet5
from model.resnet import ResNet18


def main():
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=32, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=32, shuffle=True)
    x, y = iter(cifar_train).next()
    print('x:', x.shape, 'y:', y.shape)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #device = torch.device('cpu')
    print('device:', device)

    #model = Lenet5().to(device)
    model = ResNet18(10).to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=1e-3, params=model.parameters())
    print(model)
    for epoch in range(1000):
        model.train()
        for batchIndex, (x, label) in enumerate(cifar_train):
            # x:[b, 3, 32, 32]
            x = x.to(device)
            # label:[b]
            label = label.to(device)
            # logits:[b, 10]
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        print('epoch:', epoch, 'loss:', loss.item())

        total_correct = 0
        total_num = 0

        model.eval()
        with torch.no_grad():
            for x, label in cifar_test:
                x = x.to(device)
                label = label.to(device)
                logits = model(x)
                predict = logits.argmax(dim=1)
                total_correct += torch.eq(predict, label).float().sum()
                total_num += x.size(0)

            acc = total_correct/total_num
            print('epoch:', epoch, 'accuray:', acc)

    return

if __name__ == '__main__':
    main()
