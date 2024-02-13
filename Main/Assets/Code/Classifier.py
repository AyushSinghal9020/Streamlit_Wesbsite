import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets , transforms
from tqdm.notebook import tqdm
import wandb

train_dataset = datasets.MNIST(
    root = 'data' , 
    train = True , 
    transform = transforms.ToTensor() , 
    download = True 
)

test_dataset = datasets.MNIST(
    root = 'data' , 
    train = False , 
    transform = transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_dataset , batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset , batch_size = 64 , shuffle = False)

class NeuralNetwork(nn.Module) :
    '''
    Neural Network

    Methods

    forward : Forward Pass
    '''

    def __init__(self) : 

        super(NeuralNetwork , self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(28 * 28 , 128)
        
        self.relu = nn.ReLU()
        
        self.linear2 = nn.Linear(128 , 10)

    def forward(self , x) :
        '''
        Forward Pass

        Parameters

            1) x : Input
        '''

        x = self.flatten(x)
        
        x = self.linear1(x)
        
        x = self.relu(x)
        
        x = self.linear2(x)
        
        return x

model = NeuralNetwork()

wandb.config = {
    'Epoch' : 5 , 
    'Lr' : 0.01
}

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters() , lr = 0.01)  

num_epochs = 5

wandb.watch(model)

for epoch in range(num_epochs):

    for images, labels in tqdm(train_loader , total = len(train_loader)):

        outputs = model(images)
        loss = criterion(outputs, labels)

        wandb.log({'Loss' : loss})

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

wandb.finish()