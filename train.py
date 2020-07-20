# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 05:01:42 2020

@author: lenovo
"""

#Imports here
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
import argparse
import classify_func



ap = argparse.ArgumentParser(description='Train.py')




ap.add_argument('--save_check', dest="sav_check", action="store", default="./amar_checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--Probablity_dropout', dest = "Probablity_dropout", action = "store", default = 0.45)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=409)
ap.add_argument('--no_of_prints',type = int , dest="no_of_points",action="store", default = 30)
ap.add_argument('--batch_size', type =int,dest = "batch_size",action="store", default = 32)


pa = ap.parse_args()
checkpointpath = pa.save_dir
lr = pa.learning_rate
m_name = pa.arch
pro = pa.Probablity_dropout
hidden_layer = pa.hidden_units
epoch = pa.epochs
print_every=pa.no_of_prints
size= pa.batch_size

def main():
    train_loaders,valid_loaders,test_loaders,image_train_datasets = classify_func.transform_function(size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    
    model,criterion,optimizer = classify_func.load_pre_trained(m_name,hidden_layer,pro,lr)

    classify_func.train_val_model(model,train_loaders,valid_loaders,epoch,print_every)

    classify_func.test_time(model,test_loaders)

    classify_func.save_checkpoint(model,checkpointpath)

if __name__== "__main__":
    main()