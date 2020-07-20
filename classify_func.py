# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 03:50:56 2020

@author: lenovo
"""
# Imports here
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torchvision import datasets,transforms ,models
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import json
import time
%matplotlib inline
%config InlineBackend.figure_format = 'retina'



# TODO: Define your transforms for the training, validation, and testing sets

def transform_function(size):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir,transform=train_transform)
    valid_image_datasets = datasets.ImageFolder(valid_dir,transform=valid_transform)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_image_datasets,batch_size = size,shuffle = True)
    valid_loaders = torch.utils.data.DataLoader(valid_image_datasets,batch_size = size,shuffle = False)
    test_loaders = torch.utils.data.DataLoader(test_image_datasets,batch_size = size,shuffle = False)
    
    
    return train_loaders,valid_loaders,test_loaders,train_image_datasets


##Label mapping

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
##checking Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

def load_pre_trained(m_name,hidden_layer,pro,lr):
    if m_name=='vgg16':
        model = models.vgg16(pretrained=True)
        
    elif m_name == 'resnet50':
        
        model = models.resnet50(pretrained=True)
    else:
        print("PLease select from either vgg16 model or resnet50-------No other models are allowed")
        
    # modify classifier
    for param in model.parameters():
        param.requires_grad=False
    
    classifier = nn.Sequential(nn.Linear(25088,hidden_layer),nn.ReLU(),nn.Dropout(p=pro),nn.Linear(hidden_layer,102),nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model.to(device);
    return model ,criterion,optimizer



def train_val_model(model,train_loaders,valid_loaders,epoch,print_every):
    
    epochs = epoch
    step = 0
    runn_loss = 0
    print_every = print_every
    for e in range(epochs):
        for images, labels in train_loaders:
            step+=1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
            images,labels = images.to(device) , labels.to(device)
        
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
        
            runn_loss+= loss.item()
        
            if step % print_every == 0:
                model.eval()
                valid_loss=0
                accuracy = 0
                with torch.no_grad():
                
                    for images, labels in valid_loaders:
                        images,labels = images.to(device),labels.to(device)
                
                        logps = model(images)
                        va_loss = criterion(logps,labels)
                        valid_loss += va_loss.item()
                
                        ps = torch.exp(logps)
                
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {e+1}/{epochs}.. "
                f"Train loss: {runn_loss/print_every:.3f}.. "
                f"Vlidation loss: {valid_loss/len(valid_loaders):.3f}.. "
                f"Valid accuracy: {accuracy/len(valid_loaders):.3f}")
                runn_loss = 0
                model.train()
                

###### TODO: Do validation on the test set
def test_time(model,test_loaders):
    model.eval()
    test_loss=0
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loaders:
            images,labels = images.to(device),labels.to(device)
    
            testing_log = model(images)
            t_loss =  criterion(testing_log,labels)
            test_loss+=t_loss.item()
        
            ps = torch.exp(testing_log)
            top_p,top_class = ps.topk(1,dim=1)
        
            equal = top_class == labels.view(*top_class.shape)
        
            test_accuracy+= torch.mean(equal.type(torch.FloatTensor)).item()
    print(f"test loss = {test_loss/len(test_loaders):.3f},,,"
          f"test accuracy = {test_accuracy*100/len(test_loaders):.3f}%")
        
    


# TODO: Save the checkpoint 
def save_checkpoint(model,checkpointpath):
    model.class_to_idx = train_image_datasets.class_to_idx
    checkpoint = {'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict(),
             'optimizer_state':optimizer.state_dict(),
             '}

    torch.save(checkpoint,checkpointpath )


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(pathfile):
    checkpoint = torch.load(pathfile)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    image_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    pill_transform = image_transform(pil_image)
    
    
    return pill_transform

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img = process_image(image_path).unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(img.cuda())
        
    ps = torch.exp(output)
    
    top_prob, top_class = ps.topk(topk)
    top_prob=np.array(top_prob)
    prob = [pro for pro in top_prob[0]]
    top_class = np.array(top_class)
    classes = [cat_to_name[str(index + 1)] for index in top_class[0]]
    return prob,top_class, classes


# TODO: Display an image along with the top 5 classes
def display_class(image_path,model,k=5):
    prob,top_class, classes = predict(image_path,model)
    
    plt.barh(classes,prob)
    plt.xlabel = ("Flowers Predicted")
    
    plt.ylabel = ("Probablity of Flowers")
    plt.title = ("Image Classification for Flower")
    
    
    
    imshow(process_image("flowers/test/46/image_01077.jpg"))
    
