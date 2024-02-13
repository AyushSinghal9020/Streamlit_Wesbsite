import os 
from tqdm.notebook import tqdm
from PIL import Image

import torch 
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
import wandb

import torch.nn.functional as F
from datasets import load_dataset


image_size = 28 
batch_size = 128
stats = (0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5)

dataset = load_dataset('mnist')

train_images = dataset['train']['image']
transform = transforms.ToTensor()
train_images = [
    transform(image) 
    for image 
    in tqdm(train_images , total = len(train_images) , desc = 'Transforming Images')
]
train_dl = torch.utils.data.DataLoader(train_images , batch_size = batch_size , shuffle = True)

def get_default_device() : 
    """
    Picks GPU if available else CPU

    Returns
    -------
    """
    if torch.cuda.is_available() : return torch.device('cuda')
    else : return torch.device('cpu')

def to_device(data , device) :
    """
    Move tensor(s) to chosen device

    Returns
    -------
    """
    if isinstance(data , (list , tuple)) : return [to_device(x , device) for x in data]
    return data.to(device , non_blocking = True)

class DeviceDataLoader() :
    """
    Wrap a dataloader to move data to a device
    """
    def __init__(self , dl , device) : 
        self.dl = dl
        self.device = device

    def __iter__(self) : 
        """
        Yield a batch of data after moving it to device
        """
        for b in self.dl : 
            yield to_device(b , self.device)

    def __len__(self) : 
        """
        Number of batches
        """
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl , device)

discriminator = nn.Sequential(  

    nn.Conv2d(1 , 64 , kernel_size = 4 , stride = 2 , padding = 1 , bias = False) ,
    nn.BatchNorm2d(64) ,
    nn.LeakyReLU(0.2 , inplace = True) ,

    nn.Conv2d(64 , 128 , kernel_size = 4 , stride = 1 , padding = 0 , bias = False) ,
    nn.BatchNorm2d(128) ,
    nn.LeakyReLU(0.2 , inplace = True) ,

    nn.Conv2d(128 , 256 , kernel_size = 4 , stride = 1 , padding = 1 , bias = False) , 
    nn.BatchNorm2d(256) ,
    nn.LeakyReLU(0.2 , inplace = True) ,

    nn.Conv2d(256 , 512 , kernel_size = 4 , stride = 2 , padding = 1 , bias = False) , 
    nn.BatchNorm2d(512) ,
    nn.LeakyReLU(0.2 , inplace = True) ,

    nn.Conv2d(512 , 1 , kernel_size = 4 , stride = 2 , padding = 0 , bias = False) , 

    nn.Flatten() ,
    nn.Sigmoid()
)

discriminator = to_device(discriminator , device)

latent_size = 128
fixed_latent = torch.randn(28, latent_size, 1, 1)

generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size , 512 , kernel_size = 4 , stride = 2, padding = 0 , bias = False) ,
    nn.BatchNorm2d(512) ,
    nn.ReLU(True) ,

    nn.ConvTranspose2d(512 , 256 , kernel_size = 4 , stride = 2 , padding = 1 , bias = False) ,
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256 , 128 , kernel_size = 4 , stride = 1 , padding = 0 , bias = False) ,
    nn.BatchNorm2d(128) ,
    nn.ReLU(True) ,

    nn.ConvTranspose2d(128 , 64 , kernel_size = 4 , stride = 1 , padding = 0 , bias = False) ,
    nn.BatchNorm2d(64) ,
    nn.ReLU(True) ,

    nn.ConvTranspose2d(64 , 1 , kernel_size = 4 , stride = 2 , padding = 1 , bias = False) ,
    nn.Tanh()
)

generator = to_device(generator , device)

sample_dir = 'generated'
os.makedirs(sample_dir , exist_ok = True)

def save_samples(index , latent_tensors) :
    """
    Saves a grid of generated images to file

    Args 
        1) index : int :
            Index of the current batch

        2) latent_tensors : Tensor :
            Input to the generator

    Returns
    -------
    """

    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images , os.path.join(sample_dir , fake_fname) , nrow = 10)
    print('Saving' , fake_fname)

def fit(epochs , lr , start_idx = 1) : 
    """
    Train the GAN

    Args
        1) epochs : int :
            Number of epochs

        2) lr : float :
            Learning rate

        3) start_idx : int :
            Epoch to start at (for logging)
    
    Returns
    -------
    """
    
    torch.cuda.empty_cache()
    
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    opt_d = torch.optim.Adam(discriminator.parameters( ), lr = lr , betas = (0.5 , 0.999))
    opt_g = torch.optim.Adam(generator.parameters() , lr = lr , betas = (0.5 , 0.999))
    
    wandb.watch(generator , discriminator)
    
    for epoch in range(epochs) :
    
        for real_images in tqdm(train_dl) :
            
            opt_d.zero_grad()

            real_preds = discriminator(real_images)
            real_targets = torch.ones(real_images.size(0) , 1 , device = device)
            real_loss = F.binary_cross_entropy(real_preds , real_targets)
            real_score = torch.mean(real_preds).item()

            latent = torch.randn(batch_size , latent_size , 1 , 1 , device = device)
            fake_images = generator(latent)

            fake_targets = torch.zeros(fake_images.size(0) , 1 , device = device)
            fake_preds = discriminator(fake_images)
            fake_loss = F.binary_cross_entropy(fake_preds , fake_targets)
            fake_score = torch.mean(fake_preds).item()

            loss_d = real_loss + fake_loss
            loss_d.backward()
            opt_d.step()
            
            opt_g.zero_grad()
    
            latent = torch.randn(batch_size , latent_size , 1 , 1 , device = device)
            fake_images = generator(latent)

            preds = discriminator(fake_images)
            targets = torch.ones(batch_size , 1 , device = device)
            loss_g = F.binary_cross_entropy(preds , targets)

            loss_g.backward()
            opt_g.step()
            
            wandb.log({
                'Generator Loss' : loss_g , 
                'Discriminator Loss' : loss_d , 
                'Real Score' : real_score , 
                'Fake Score' : fake_score
            })
            
        wandb.log({
            'Epoch' : epoch
        })
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        save_samples(epoch+start_idx, fixed_latent, show=False)
        Image('./generated/generated-images-{0:0=4d}.png'.format(epoch))
    
    return losses_g, losses_d, real_scores, fake_scores

lr = 0.002
epochs = 25

wandb.init(project = 'MNIST GAN')
wandb.config = {'Epoch' : 25 , 'learning_rate' : 0.002 , 'batch_size' : 128}

history = fit(epochs, lr)