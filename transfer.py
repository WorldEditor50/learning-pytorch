from random import shuffle
import torch
from torch import nn, optim
from torch.utils import DataLoader
from imageloader import ImageLoader
#from resnet import Resnet18
from torchvision.models import resnet18


class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()


    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape()[:-1])).item()
        return x.view(-1, shape)


batchsize = 32
lr = 1e-3
max_epoch = 10
device = torch.device('cuda')
torch.manual_seed(1234)
image_path = 'D:\home\dataset\pokeman',
# load data
train_db = ImageLoader(image_path, 224, mode='train')
valid_db = ImageLoader(image_path, 224, mode='validate')
test_db = ImageLoader(image_path, 224, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsize, shuffle=True, num_worker=4)
valid_loader = DataLoader(valid_db, batch_size=batchsize, num_worker=2)
test_loader = DataLoader(test_db, batch_size=batchsize, num_worker=2)

def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            predicts = logits.argmax(dim=1)
        correct += torch.eq(predicts, y).sum().float().item()
    return correct/total

def main():
    # model = Resnet18(5).to(device)
    # transfer learning
    trained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], # [b, 512, 1, 1]
        Flatten(), # [b, 512, 1, 1] => [b, 512]
        nn.Linear(512, 5)
    )
    model = model.to(device)
    optimizer = nn.optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc = 0
    best_epoch = 0
    for epoch in range(max_epoch):
        # train
        for step, (x, y) in enumerate(train_db):
            # x: [b, 3, 224, 224]
            x = x.to(device)
            # y: [b]
            y = y.to(device)
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation
        if epoch % 2:
            valid_acc = evaluate(model, valid_loader)
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'best.model')
    # test
    print('best accuracy:', best_acc, 'best epoch', best_epoch)
    model.load_state_dict(torch.load('best.model'))
    test_acc = evaluate(model, test_loader)
    print('test accuracy:', test_acc)

if __name__ == '__main__':
    main()
