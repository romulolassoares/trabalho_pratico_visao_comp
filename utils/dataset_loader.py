import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score,classification_report
from datetime import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
from glob import glob

# ========================================================================================
# Load cifar10 dataset
# ========================================================================================
def load_cifar10(n_size: int = 100, transform: Callable = None):
    if transform is None:
        raise ValueError('Transform cannot be None')
        
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainset_percentage = n_size / 100
    trainset_size = int(trainset_percentage * len(train_dataset))
    trainset = torch.utils.data.Subset(train_dataset, range(trainset_size))
    
    
    testset_percentage = n_size / 100
    testset_size = int(testset_percentage * len(test_dataset))
    testset = torch.utils.data.Subset(test_dataset, range(testset_size))

    return trainset, testset


# ========================================================================================
# Load custom dataset
# ========================================================================================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data = data_path
        self.label = [int(p.split('/')[-2] == 'santa') for p in data_path]
        self.data_len = len(self.data)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        image = Image.open(self.data[index], mode='r')
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        self.label[index] = np.array(self.label[index])
        return image, torch.from_numpy(self.label[index])
