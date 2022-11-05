import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import autograd
import os
import time
from tqdm import tqdm
import dataset
from dataset import AnimeDataset
from dataset import to_image
from model import Generator, Discriminator, gradientPenalty
num_epoch = 1024
z_dim = 100
img_path = './anime-faces'
generate_img_path = './animewgan_img'
model_path = './anime_model'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4




def main():
    print("Device:", DEVICE)
    if not os.path.exists(generate_img_path):
        os.mkdir(generate_img_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # load data
    dataset = AnimeDataset(img_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # model
    G = Generator(z_dim, img_channels=3, feature_channels=16).to(DEVICE)
    D = Discriminator(img_channels=3, feature_channels=16).to(DEVICE)

    # load parameter
    G.weight_init()
    D.weight_init()
    #G.load_state_dict(torch.load("./anime-model/wgan_generator.pth"))
    #D.load_state_dict(torch.load("./anime-model/wgan_discriminator.pth"))

    # optimizer
    optimizerG = torch.optim.RMSprop(G.parameters(), lr = learning_rate)
    optimizerD = torch.optim.RMSprop(D.parameters(), lr = learning_rate)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)

    # start training
    one = torch.FloatTensor([1]).to(DEVICE)
    mone = -1*one

    z_sample = torch.randn(100, z_dim).to(DEVICE)
    z_sample = z_sample.reshape(100, z_dim, 1, 1)
    for e, epoch in enumerate(range(num_epoch)):
        for i, data in enumerate(tqdm(dataloader, desc='epoch={}'.format(e))):
            # train discriminator
            imgs = data
            batchsize = imgs.size(0)
            imgr = imgs.to(DEVICE)
            for param in D.parameters():
                param.requires_grad = True
            
            optimizerD.zero_grad()
            # train with real image
            yr = D(imgr)
            lossr = yr.mean(0).view(1)
            lossr.backward(one)
            # train with fake image
            z = torch.randn(batchsize, z_dim).to(DEVICE)
            z = z.reshape(batchsize, z_dim, 1, 1)
            imgf = G(z).detach()
            yf = D(imgf)
            # discriminator loss
            lossf = yf.mean(0).view(1)
            lossf.backward(mone)

            lossD = lossf - lossr
            optimizerD.step()
            # clip weight
            for param in D.parameters():
                param.data.clamp_(-0.01, 0.01)

            for param in D.parameters():
                param.requires_grad = False

            # train generator
            optimizerG.zero_grad()
            z = torch.randn(batchsize, z_dim).to(DEVICE)
            z = z.reshape(batchsize, z_dim, 1, 1)
            imgf = G(z)
            yf = D(imgf)
            lossG = -torch.mean(yf).view(1)
            lossG.backward(one)
            optimizerG.step()

            # scheduler
            schedulerG.step()
            schedulerD.step()

        # generate image
        with torch.no_grad():
            img = to_image(G(z_sample)).to(DEVICE)
            postfix = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            filename = os.path.join('animewgan_img', f'img_{postfix}.jpg')
            torchvision.utils.save_image(img, filename, nrow=10)
            print(f'| generate sample to {filename}')
            torch.save(G.state_dict(), os.path.join(model_path, 'wgan_generator_{}.pth'.format(postfix)))
            torch.save(D.state_dict(), os.path.join(model_path, 'wgan_discriminator_{}.pth'.format(postfix)))
    return


if __name__ == '__main__':
    main()