import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


"""
              channels
    1    64   64                                                                        128  64   64   2
    |    |    |                                                                         |    |    |    |
    |    |    |                                                                         |    |    |    |
    | -> | -> |                                   =>  copy and crop                     | -> | -> | -> |
    |conv|    |                                                                         |    |    |    |
    |    |    |                                                                         |    |    |    |

              |  down sample (encoder)                                                  ^  up sample (decoder)
              V                                                                         |
                   128  128                                                       128
              |    |    |                                                     |    |    |
              | -> | -> |                         =>                          | -> | -> |
              |    |    |                                                     |    |    |
              |    |    |                                                     |    |    |

                        |                                                     ^
                        V                                                     |
                             256  256                                    256
                        |    |    |                                 |    |    |
                        | -> | -> |               =>                | -> | -> | 
                        |    |    |                                 |    |    |

                                  |                                 ^
                                  V                                 |
                                       512 512                512
                                  | -> | -> |     =>      | -> | -> |
                                  |    |    |             |    |    |

                                            |             ^
                                            V             |
                                                   1024   1024
                                            |  ->  |  ->  |

                                

"""
# input image
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]) -> None:
        super(UNET, self).__init__()

        self.downSample = nn.ModuleList()
        self.upSample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down sample part
        for feature in features:
            self.downSample.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up sample part
        for feature in reversed(features):
            self.upSample.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upSample.append(DoubleConv(feature*2, feature))
        
        # bottleneck [512, 1024, 1024]
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        downSampleResults = []
        # down sample
        for block in self.downSample:
            x = block(x)
            downSampleResults.append(x)
            x = self.pool(x)
        # bottle neck
        x = self.bottleneck(x)
        # up sample
        downSampleResults = downSampleResults[::-1]
        for index in range(0, len(self.upSample), 2):
            x = self.upSample[index](x)
            result = downSampleResults[index//2]
            # resize
            if x.shape != result.shape:
                x = TF.resize(x, size=result.shape[2:])

            x_up = torch.cat((result, x), dim=1)
            x = self.upSample[index + 1](x_up)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    predits = model(x)
    print(predits.shape)
    print(x.shape)
    assert predits.shape == x.shape

if __name__ == '__main__':
    test()
        