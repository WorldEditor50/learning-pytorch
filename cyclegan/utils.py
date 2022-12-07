import torch
import torchvision
import os 
import random 
import numpy as np 
import config

def save_checkpoint(model, optimizer, filename="cyclegan_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    return 

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    torch.load(checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 

    return 


def save_everthing(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed) 
    np.random(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return 


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)
    
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return

def save_predictions_as_imgs(loader, model, folder="save_images/", device="cuda"):
    model.eval()

    for index, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{index}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/y_{index}.png")

    model.train()
    return
