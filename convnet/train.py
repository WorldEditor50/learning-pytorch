#!/usr/bin/env python
# encoding: utf-8

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from model import FooNN
import time
# get data
train_data = torchvision.datasets.CIFAR10(root="test_dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="test_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
# load data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("train with gpu")
else:
    device = torch.device("cpu")
    print("train with cpu")

# set train parameter
total_train_step = 0
total_test_step = 0
total_loss = 0
total_accuracy = 0
epoch = 10
# tensor board: localhost:6006
writer = SummaryWriter("./log_train")
if __name__ == '__main__':
    print("train data len: {}".format(train_data_size))
    print("test data len: {}".format(test_data_size))
    # test nn
    foo = FooNN()
    foo = foo.to(device)
    input = torch.ones((64, 3, 32, 32))
    output = foo(input)
    print(output.shape)
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    learingRate = 1e-2
    optimizer = torch.optim.SGD(foo.parameters(), lr=learingRate)
    # train
    foo.train()
    for i in range(epoch):
        print("------epoch :{}-------".format(i))
        start_time = time.time()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = foo(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("cost time:{}".format(end_time - start_time))
                print("train step:{}, loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        # test
        foo.eval()
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = foo(imgs)
                # test loss
                test_loss = loss_fn(outputs, targets)
                total_loss = total_loss + test_loss.item()
                # accuracy
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
        print("total loss: {}".format(total_loss))
        print("accuracy: {}".format(total_accuracy/test_data_size))
        writer.add_scalar("test_loss", total_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
        total_test_step = total_test_step + 1
        # save model
        torch.save(foo, "foo_{}.pth".format(i))
        # recommend method: torch.save(foo.state_dict(), "foo_{}.pth".format(i))
        print("saved model.")
    writer.close()
