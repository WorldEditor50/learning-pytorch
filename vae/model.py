import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class VAE(nn.Module):

    def __init__(self, img_size, h_dim=200, z_dim=20) -> None:
        super().__init__()
        # encoder
        self.fromImg = nn.Linear(img_size, h_dim)
        self.map2mu = nn.Linear(h_dim, z_dim)
        self.map2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.fromZ = nn.Linear(z_dim, h_dim)
        self.map2Img = nn.Linear(h_dim, img_size)

        self.relu = nn.ReLU()


    def encoder(self, x):
        h = self.relu(self.fromImg(x))
        mu = self.map2mu(h)
        sigma = self.map2sigma(h)
        return mu, sigma


    def decoder(self, z):
        h = self.relu(self.fromZ(z))
        return torch.sigmoid(self.map2Img(h))


    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.rand_like(sigma)
        z_reparameterized = mu + sigma*epsilon
        x_reconstructed = self.decoder(z_reparameterized)
        return x_reconstructed, mu, sigma



if __name__ == "__main__":
    x = torch.randn(4, 28*28)
    model = VAE(img_size=784)
    y, mu, sigma = model(x)
    print(y.shape)
    print(mu.shape)
    print(sigma.shape)
