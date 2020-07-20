# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 05:34:32 2020

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



ap = argparse.ArgumentParser(description='Predict.py')




ap.add_argument('--load_check', dest="load_check", action="store", default="./amar_checkpoint.pth")
ap.add_argument('--image_file', dest="image_file", action="store", default="./flowers/test/1/image_06743.jpg")



pa = ap.parse_args()
pathfile = pa.load_check
image_path = pa.image_file


def main():
    with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    model = classify_func.load_checkpoint(pathfile)
    
    classify_func.predict(image_path,model)
    
    classify_func.display_class(image_path,model)
    
    
if __name__== "__main__":
    main()


