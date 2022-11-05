import torch
from torch.utils.data import DataLoader
from dataset import AnimeDataset
from model import Generator, Discriminator


def test():
    img_path = './anime-faces'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_dim = 100

    print("Device:", device)
    
    # load data
    dataset = AnimeDataset(img_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    G = Generator(z_dim, img_channels=3, feature_channels=16).to(device)
    D = Discriminator(img_channels=3, feature_channels=16).to(device)

    x = torch.randn(32, z_dim).to(device)
    x = x.reshape(32, z_dim, 1, 1)
    print("x:", x.shape)
    y = G(x)
    print("y:", y.shape)

    # test discrimnator [32, 3, 64, 64])
    for i, data in enumerate(dataloader):
        img = data
        img = img.to(device)
        print("img:", img.shape)
        yr = D(img)
        print("yr:", yr.shape)
        break
    return

if __name__ == '__main__':
    test()