#!/usr/bin/env python
# encoding: utf-8

import torch
from PIL import Image
import torchvision
from torch import nn
from torchvision import transforms
from model import FooNN

#image
img_path = "/home/eigen/Pictures/dog.jpeg"

if __name__ == '__main__':
    image = Image.open(img_path)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor()])
    image = transform(image)
    # load model
    model = torch.load("foo_9.pth", map_location=torch.device("cpu"))
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image)
    print("predict", output)
    print("index: {}".format(output.argmax(1)))
