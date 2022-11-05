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
    
    if not os.path.exists("animewgangp_img"):
        os.mkdir("animewgangp_img")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # load data
    dataset = AnimeDataset(img_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # model
    G = Generator(z_dim, img_channels=3, feature_channels=16).to(DEVICE)
    D = Discriminator(img_channels=3, feature_channels=16).to(DEVICE)

    # load parameter
    #G.weight_init()
    #D.weight_init()
    G.load_state_dict(torch.load("./anime_model/wgangp_generator.pth"))
    D.load_state_dict(torch.load("./anime_model/wgangp_discriminator.pth"))

    # optimizer
    optimizerG = torch.optim.Adam(G.parameters(), lr = learning_rate, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(D.parameters(), lr = learning_rate, betas=(0.0, 0.9))

    # start training
    z_sample = torch.randn(100, z_dim).to(DEVICE)
    z_sample = z_sample.reshape(100, z_dim, 1, 1)
    for e, epoch in enumerate(range(num_epoch)):
        for i, data in enumerate(tqdm(dataloader, desc='epoch={}'.format(e))):
            # train discriminator
            imgs = data
            batchsize = imgs.size(0)
            imgr = imgs.to(DEVICE)

            # train with real image
            yr = D(imgr)
            # train with fake image
            z = torch.randn(batchsize, z_dim).to(DEVICE)
            z = z.reshape(batchsize, z_dim, 1, 1)
            imgf = G(z).detach()
            yf = D(imgf)
            # gradient penalty
            gp = gradientPenalty(D, imgr, imgf, DEVICE)
            # discriminator loss
            lossD = torch.mean(yf) - torch.mean(yr) + 10*gp
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            # train generator
            if i%5 == 0:
                optimizerG.zero_grad()
                z = torch.randn(batchsize, z_dim).to(DEVICE)
                z = z.reshape(batchsize, z_dim, 1, 1)
                imgf = G(z)
                yf = D(imgf)
                lossG = -torch.mean(yf)
                lossG.backward()
                optimizerG.step()

        # generate image
        with torch.no_grad():
            img = to_image(G(z_sample)).to(DEVICE)

            postfix = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            filename = os.path.join('animewgangp_img', f'img_{postfix}.jpg')
            torchvision.utils.save_image(img, filename, nrow=10)
            print(f'| generate sample to {filename}')
            torch.save(G.state_dict(), os.path.join(model_path, 'wgangp_generator.pth'))
            torch.save(D.state_dict(), os.path.join(model_path, 'wgangp_discriminator.pth'))
    return

if __name__ == '__main__':
    main()