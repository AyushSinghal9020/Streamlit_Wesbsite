import os  
import pandas
import json
import torch 
import numpy as np
from PIL import Image
import torch.nn as nn 
from tqdm.notebook import tqdm
from torch.utils.data import Dataset , DataLoader

from torchvision.utils import save_image
import wandb

class hubmapDataset(Dataset):
    
    def __init__(self, image_dir, labels_file , augments = False):
        
        with open(labels_file, 'r') as json_file:
            self.json_labels = [json.loads(line) for line in json_file]

        self.image_dir = image_dir
        self.augments = augments

    __len__ = lambda self : len(self.json_labels)    
        
    def __getitem__(self, idx):
        
        image_path = os.path.join(self.image_dir, f"{self.json_labels[idx]['id']}.tif")
        image = Image.open(image_path)
        
        mask = np.zeros((512, 512), dtype=np.float32)

        for annot in self.json_labels[idx]['annotations']:

            cords = annot['coordinates']
            
            if annot['type'] == "blood_vessel":
                
                for cord in cords:
                    
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    
                    mask[rr, cc] = 1

        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) 
        mask = torch.tensor([mask], dtype=torch.float32)

        return image, mask

train_dataset = hubmapDataset(
    image_dir = '/kaggle/input/hubmap-hacking-the-human-vasculature/train' , 
    labels_file = '../input/hubmap-hacking-the-human-vasculature/polygons.jsonl'
)

train = DataLoader(train_dataset)

class UNet(nn.Module) : 

    def __init__(self) : 

        super(UNet , self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.MaxPool2d(kernel_size = 2 , stride = 2)
        )

        self.mid_encoder = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.MaxPool2d(kernel_size = 2 , stride = 2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128 , 64 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace = True) ,
            nn.ConvTranspose2d(64 , 64 , kernel_size = 2 , stride = 2 , output_padding = 0) , 
            nn.ConvTranspose2d(64 , 1 , kernel_size = 2 , stride = 2 , output_padding = 0)
        )
        
        self.fact = nn.Sigmoid()

    def forward(self, inps):

        inps = self.encoder(inps)
        inps = self.mid_encoder(inps)
        inps = self.decoder(inps)
        inps = self.fact(inps)

        return inps

import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.mid_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, output_padding=0) , 
            nn.ConvTranspose2d(64 , 1 , kernel_size = 2 , stride = 2 , output_padding = 0)
        )
        
        self.fact = nn.Sigmoid()

    def forward(self, x):

        x1 = self.encoder(x)
        x2 = self.mid_encoder(x1)
        x3 = self.decoder(x2)
        
        x3 = self.fact(x3)

        return x3

un = UNet().to('cuda')
un_optim = torch.optim.SGD(un.parameters() , lr = 0.0001)

vn = VNet().to('cuda')
vn_optim = torch.optim.SGD(vn.parameters() , lr = 0.0001)

criterion = nn.BCELoss()

wandb.watch((un , vn) , criterion)

un_losses = []
vn_losses = []

for inputs , labels in tqdm(train , total = 1633) :
    
    inputs = inputs.to('cuda')
    labels = labels.to('cuda')
    
    un_out = un(inputs)
    vn_out = vn(inputs)
    
    un_loss = criterion(un_out , labels)
    vn_loss = criterion(vn_out , labels)
    
    un_losses.append(un_loss)
    vn_losses.append(vn_loss)
    
    wandb.log({
        'Vn_Loss' : vn_loss , 
        'un_loss' : un_loss
    })
    
    un_loss.backward()
    vn_loss.backward()

    un_optim.step()
    vn_optim.step()
    
    un_optim.zero_grad()
    vn_optim.zero_grad()
    
    torch.cuda.empty_cache()