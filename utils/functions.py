import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  #
from typing import Callable
from datetime import datetime
from classes.AlexNet import AlexNet

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
# Plot confusion matrix
# ========================================================================================
def plot_confusion_matrix(model, dataloader, classes, device, type):
    model.eval()
    model = model.to(device)  # Mova o modelo para o dispositivo correto
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        tqdm_dataloader = tqdm(dataloader, desc='Generating Confusion Matrix', dynamic_ncols=True)
        for inputs, labels in tqdm_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mova os dados para o dispositivo correto
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    tqdm_dataloader.close()

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    formatted_datetime = datetime.now().strftime("%d_%m_%H_%M")
    plt.savefig(f'./images/cm_{formatted_datetime}_{type}.png')


