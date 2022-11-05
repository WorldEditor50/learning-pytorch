
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, img_path) -> None:
        super().__init__()
        self.fnames = [img_path + '/' + img for img in os.listdir(img_path)]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.num_samples = len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def to_image(x):
    img = (x + 1)/2
    return img