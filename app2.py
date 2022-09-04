import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #For ploting
import torch    #Pytorch
import os
from flask import Flask, request
from flask_cors import CORS

import torchvision.transforms as tt #To apply transformations to the dataset, augmenting it and transforming it to a tensor.
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder #Load dataset

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tqdm.notebook import tqdm

import PIL

# ---------------------------------------------------------------------------------------

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# ------------------------------------------------------------------------------------------





train_tf = tt.Compose([         
    tt.ColorJitter(brightness=0.2),
    tt.Resize(size=(150,150)),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(5),
    tt.ToTensor(),            
])

#Transformations aplied to test DS.
test_tf= tt.Compose([   
    tt.Resize(size=(150,150)),
    tt.ToTensor(),
])


test_dir = 'seg_test/seg_test'
test_ds = ImageFolder(test_dir,test_tf)

batch_size = 254

test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=3,
    shuffle=False,
    pin_memory=True
)
# -----------------------------------------------------------------------------------------------

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
# ---------------------------------------------------------------------------------------------------

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
# ---------------------------------------------------------------------------------------------------

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            (epoch+1), result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
# ---------------------------------------------------------------------------------------------------------------------

class Resnet34(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
# --------------------------------------------------------------------------------------------------------------
device = get_default_device() #Getting the device

state = torch.load('checkpoint.pth')

model2 = to_device(Resnet34(), device)

model2.load_state_dict(state)
# device

# classes = ['cheetah', 'lion', 'tiger']

def show_sample(img, target):
    plt.imshow(img.permute(1, 2, 0))
    print('Labels:', target)

test_transforms = tt.Compose([   
    tt.Resize(255),
    tt.CenterCrop(224),
    tt.ToTensor(),
])
   
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch Tensor
    '''
    im = PIL.Image.open(image)
    return test_transforms(im)
    
def predict_image(img, model):
    # Convert to a batch of 1
    img_pros = process_image(img)
    img = img_pros.view(1, 3, 224, 224)
    xb = to_device(img, device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    # show_sample(img,test_ds.classes[preds[0].item()])
    return test_ds.classes[preds[0].item()-2]
# ------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

@app.route('/pred', methods=['POST'])
def pred():
    img = request.files['img']
    a = predict_image(img,model2)
    return a

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)