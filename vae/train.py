import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn 
from model import VAE 
from torchvision import transforms 
from torchvision.utils import save_image 
from torch.utils.data import DataLoader 

# configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 784 
H_DIM = 200
Z_DIM = 20 
NUM_EPOCHS = 30 
BATCH_SIZE = 64 
LEARNING_RATE = 1e-4

mnist = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=mnist, batch_size=BATCH_SIZE, shuffle=True)


def inference(model, digit, num_examples):
    images = []
    index = 0
    for x, y in mnist:
        if y == index:
            images.append(x)
            index+=1
        if index == 10:
            break

    encodeing_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encoder(images[d].view(1, 784))
        encodeing_digit.append((mu, sigma))
    
    mu, sigma = encodeing_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_example_{example}.png")
def main():
    # train
    model = VAE(IMG_SIZE, H_DIM, Z_DIM).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss(reduction="sum")
    for epoch in range(NUM_EPOCHS):

        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            x = x.to(DEVICE).view(x.shape[0], IMG_SIZE)
            x_reconstructed, mu, sigma = model(x)

            reconstructed_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstructed_loss + kl_div 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    # inference
    model = model.to("cpu")
    for i in range(10):
        inference(model, i, num_examples=1)
    
if __name__ == "__main__":
    main()
