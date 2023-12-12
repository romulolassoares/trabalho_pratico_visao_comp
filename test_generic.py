import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from classes.AlexNet import AlexNet
from classes.Vgg_pytorch import Vgg
from utils.functions import *
import os
import json
from datetime import datetime

# Carregar configurações do YAML
with open('./test_infos.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)


basic_infos = {
    'cifar': {
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'santa': {
        'classes': ['Santa', 'Not Santa']
    }
}

for test_config in config['tests']:
    NUM_EPOCHS = BATCH_SIZE = MOMENTUM = LR_DECAY = LR_INIT = IMAGE_DIM = NUM_CLASSES = 0
    CIFAR10_CLASSES = []
    
    print(f"Test: {test_config['name']}")
    dataset = test_config['dataset']
    
    # Número de épocas
    NUM_EPOCHS = test_config['num_epochs']
    # Tamanho do lote em cada interação
    BATCH_SIZE = test_config['batch_size']
    # Parâmetro de convergência
    MOMENTUM = test_config['momentum']
    # Taxa de decaimento
    LR_DECAY = test_config['lr_decay']
    # Taxa de aprendizado inicial
    LR_INIT = test_config['lr_init'] 
    # Dimensão das imagens
    IMAGE_DIM = 227 if test_config['model'] == 'alexnet' else 32
    # Lista de classes do dataset
    CIFAR10_CLASSES = basic_infos[dataset]['classes']
    # Total de classes
    NUM_CLASSES = len(CIFAR10_CLASSES)
    
    print(NUM_EPOCHS, BATCH_SIZE,IMAGE_DIM)


    # ========================================================================================
    # Treinamento
    # ========================================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Defina as transformações para redimensionar e normalizar as imagens
    transform = transforms.Compose([transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'cifar':
        trainset, testset = load_cifar10(n_size=100, transform=transform)
        # Defina o dataloader
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_mother_path = './data/is that santa/train'
        test_mother_path = './data/is that santa/train'
        train_image_path = glob(os.path.join(train_mother_path, '*', '*'))
        test_image_path = glob(os.path.join(test_mother_path, '*', '*'))
        trainData = CustomDataset(train_image_path, transform=transform)
        testData = CustomDataset(test_image_path, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainData,batch_size=BATCH_SIZE,shuffle=True)
        testloader = torch.utils.data.DataLoader(testData,batch_size=BATCH_SIZE,shuffle=False)

    
    # Instancie a AlexNet
    if test_config['model'] == 'alexnet':
        model = AlexNet(num_classes=NUM_CLASSES).to(device)
    else:
        model = Vgg(num_classes=NUM_CLASSES).to(device)

    # Defina a função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=LR_DECAY)

    model_trained, metrics = make_train(model, NUM_EPOCHS, optimizer,criterion,trainloader, testloader, device)


    metrics_df = pd.DataFrame(metrics)
    
    

    # ========================================================================================
    # Salvando os arquivos de resultado
    # ========================================================================================
    path = './test/results'
    base_name = f'e{NUM_EPOCHS}b{BATCH_SIZE}class{NUM_CLASSES}'
    # Adicionando a data e hora ao nome da pasta
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f'{path}/{test_config["name"]}_{base_name}_{current_time}'

    os.makedirs(folder, exist_ok=True)


    info = {
    'epoca': NUM_EPOCHS,
    'lote': BATCH_SIZE,
    'momento': MOMENTUM,
    'decaimento': LR_DECAY,
    'tx_inical': LR_INIT,
    'dimensao': IMAGE_DIM,
    'classes': CIFAR10_CLASSES,
    'total_classes': NUM_CLASSES
    }

    # Plote gráficos de perda
    plt.figure(figsize=(12, 6))  # Create the first figure
    plt.plot(metrics_df['train_loss'], label='Training Loss')
    plt.plot(metrics_df['test_loss'], label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.savefig(f'{folder}/loss_over_time.png')

    plt.figure(figsize=(12, 6))  # Create the first figure
    fig, ax, report = evaluate_model(model_trained, testloader, CIFAR10_CLASSES, device)
    plt.savefig(f'{folder}/confusion_matrix.png')
    report_df = pd.DataFrame(report).T


    with open(f'{folder}/info.json', 'w') as json_file:
        json.dump(info, json_file, indent=2)


    report_df.to_csv(f'{folder}/report.csv')

    
    torch.cuda.empty_cache()