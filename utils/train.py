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

# Função para realizar o treinamento dos modelos (Pytorch)
def train(model, optimizer, criterion, train_loader, epoch, device):
    model.train()
    running_loss = 0.0
    
    tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}, Training', dynamic_ncols=True)
    for i, data in enumerate(tqdm_train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tqdm_train_loader.set_postfix({'loss': running_loss / (i + 1)})

    train_loss = running_loss / len(train_loader)
    tqdm_train_loader.close()
    
    return train_loss
    
# Função para realizar os testes do modelo
def test(model, criterion, test_loader, epoch, device):
    model.eval()
    running_test_loss = 0.0
    predicted_labels = []
    true_labels = []

    tqdm_test_loader = tqdm(test_loader, desc=f'Epoch {epoch}, Testing', dynamic_ncols=True)
    with torch.no_grad():
        for data in tqdm_test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(outputs, 1)  # Obtém a classe prevista
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_loss = running_test_loss / len(test_loader)
    tqdm_test_loader.close()

    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return test_loss, accuracy, recall, f1

# ========================================================================================
# Make training
# ========================================================================================
def handle_train(model, num_epochs, optimizer, criterion, train_loader, test_loader,device):
    metrics = []
    total_steps = 0
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, epoch+1, device)
        test_loss, accuracy, recall, f1 = test(model, criterion, test_loader, epoch+1, device)

        result_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        }
        metrics.append(result_data)

        total_steps += 1

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return model, metrics


