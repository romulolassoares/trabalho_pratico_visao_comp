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


# Defina a arquitetura da AlexNet simplificada
class AlexNet(nn.Module):
    """
    Neural Network model proposed by AlexNet paper
    """
    # def __init__(self, num_classes=10):
    #     """
    #     Define the layers os AlexNet

    #     Args:
    #         num_classes (int, optional): number of classes to predict with this model. Defaults to 10.
    #     """
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Conv2d(3, 96, kernel_size=11, stride=4),
    #         nn.ReLU(),
    #         nn.LocalResponseNorm(5,0.0001, beta=0.75, k=2),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(96, 256, kernel_size=5, padding=2),
    #         nn.ReLU(),
    #         nn.LocalResponseNorm(5,0.0001, beta=0.75, k=2),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(256, 384, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(384, 384, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(p=0.5, inplace=True),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5, inplace=True),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(),
    #         nn.Linear(4096, num_classes),
    #     )
    #     self.init_bias()
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # self.init_bias()
    
    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Input through the net

        Args:
            x (Tensor): input tensor

        Returns:
            (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x
