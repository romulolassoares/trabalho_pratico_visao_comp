# ========================================================================================
# Imports
# ========================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from classes.AlexNet import AlexNet
from utils.functions import *
import os
import json
from datetime import datetime



# ========================================================================================
# Variaveis que precisam ser alteradas em cada teste
# ========================================================================================
# Número de épocas
NUM_EPOCHS = 50
# Tamanho do lote em cada interação
BATCH_SIZE = 128
# Parâmetro de convergência
MOMENTUM = 0.9 
# Taxa de decaimento
LR_DECAY = 0.0005 
# Taxa de aprendizado inicial
LR_INIT = 0.012 
# Dimensão das imagens
IMAGE_DIM = 227  
# Lista de classes do dataset
CIFAR10_CLASSES = ['Santa', 'Not Santa']
# Total de classes
NUM_CLASSES = len(CIFAR10_CLASSES)
# Porcentagem total do dataset
PERCENT = 100

# ========================================================================================
# Daqui para baixo não precisa de alterações
# ========================================================================================

# ========================================================================================
# Treinamento
# ========================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Defina as transformações para redimensionar e normalizar as imagens
transform = transforms.Compose([transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset, testset = load_cifar10(n_size=PERCENT, transform=transform)
# Defina o dataloader
# Defina o dataloader
train_mother_path = './data/is that santa/train'
test_mother_path = './data/is that santa/train'
train_image_path = glob(os.path.join(train_mother_path, '*', '*'))
test_image_path = glob(os.path.join(test_mother_path, '*', '*'))
trainData = CustomDataset(train_image_path, transform=transform)
testData = CustomDataset(test_image_path, transform=transform)
trainloader = torch.utils.data.DataLoader(trainData,batch_size=BATCH_SIZE,shuffle=True)
testloader = torch.utils.data.DataLoader(testData,batch_size=BATCH_SIZE,shuffle=False)

# Instancie a AlexNet
alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

# Defina a função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=LR_DECAY)

model_trained, metrics = make_train(alexnet, NUM_EPOCHS, optimizer,criterion,trainloader, testloader, device)


metrics_df = pd.DataFrame(metrics)

# ========================================================================================
# Salvando os arquivos de resultado
# ========================================================================================
path = './test/results'
base_name = f'e{NUM_EPOCHS}b{BATCH_SIZE}class{NUM_CLASSES}'
# Adicionando a data e hora ao nome da pasta
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f'{path}/{base_name}_{current_time}'

os.makedirs(folder, exist_ok=True)


info = {
'epoca': NUM_EPOCHS,
'lote': BATCH_SIZE,
'momento': MOMENTUM,
'decaimento': LR_DECAY,
'tx_inical': LR_INIT,
'dimensao': IMAGE_DIM,
'classes': CIFAR10_CLASSES,
'total_classes': NUM_CLASSES,
'porcentagem': PERCENT,
}

# Plote gráficos de perda
plt.plot(metrics_df['train_loss'], label='Training Loss')
plt.plot(metrics_df['test_loss'], label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.savefig(f'{folder}/loss_over_time.png')


fig, ax, report = evaluate_model(model_trained, testloader, CIFAR10_CLASSES, device)
report_df = pd.DataFrame(report).T


with open(f'{folder}/info.json', 'w') as json_file:
    json.dump(info, json_file, indent=2)


report_df.to_csv(f'{folder}/report.csv')

plt.savefig(f'{folder}/confusion_matrix.png')