import torch 
from PIL import Image
import os 
from torch.utils.data import Dataset 


class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=True) -> None:
        super().__init__()
        self.root_zebra = root_zebra 
        self.root_horse = root_horse 
        self.transform = True 

        self.zebra_images = os.listdir(root_zebra) 
        self.horse_images = os.listdir(root_horse) 
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) 
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images) 


    def __len__(self):
        return self.length_dataset 


    def __getitem__(self, index): 

        zebra_img = self.zebra_images[index%self.zebra_len] 
        horse_img = self.horse_images[index%self.horse_len] 

        zebra_path = os.path.join(self.root_zebra, zebra_img) 
        horse_path = os.path.join(self.root_horse, horse_img) 

        if self.transform: 
            augumentations = self.transform(image=zebra_img, image0=horse_img) 
            zebra_img = augumentations["image"] 
            horse_img = augumentations["image0"]

        return zebra_img, horse_img

