import torch
import torch.nn as nn
from torch import autograd

# generator
class Generator(nn.Module):

    def __init__(self, z_channels, img_channels, feature_channels) -> None:
        super(Generator, self).__init__()

        def convtransposeblock(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.net = nn.Sequential(
            # input: [b, z_channels, 1, 1]
            convtransposeblock(z_channels, feature_channels*16, 4, 2, 0),
            convtransposeblock(feature_channels*16, feature_channels*8, 4, 2, 1),
            convtransposeblock(feature_channels*8, feature_channels*4, 4, 2, 1),
            convtransposeblock(feature_channels*4, feature_channels*2, 4, 2, 1),
            # output: [b, img_channels, 64, 64]
            nn.ConvTranspose2d(feature_channels*2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def weight_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
        return

    def forward(self, x):
        y = self.net(x)
        return y
# discriminator
class Discriminator(nn.Module):

    def __init__(self, img_channels, feature_channels) -> None:
        super(Discriminator, self).__init__()

        def convblock(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2)
            )
        self.net = nn.Sequential(
            # [b, 3, 64, 64]
            nn.Conv2d(img_channels, feature_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            convblock(feature_channels, feature_channels*2, 4, 2, 1),
            convblock(feature_channels*2, feature_channels*4, 4, 2, 1),
            convblock(feature_channels*4, feature_channels*8, 4, 2, 1),
            nn.Conv2d(feature_channels*8, 1, kernel_size=4, stride=2, padding=0, bias=False)
        )

    def weight_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
        return

    def forward(self, x):
        y = self.net(x)
        return y

def gradientPenalty(D, imgr, imgf, device):
    BATCH_SIZE, C, H, W = imgr.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # Get random interpolation between real and fake samples
    img = (alpha*imgr + ((1 - alpha)*imgf)).requires_grad_(True)
    yi = D(img)
    #gradout = Variable(Tensor(imgr.shape[0], 1).fill_(1.0), requires_grad=False)
    gradout = torch.ones(imgr.size(0), 1).to(device)
    # Get gradient w.r.t. interpolates
    grad = autograd.grad(
        inputs=img,
        outputs=yi,
        grad_outputs=torch.ones_like(yi),
        create_graph=True,
        retain_graph=True
        )[0]
    grad = grad.view(grad.size(0), -1)
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
