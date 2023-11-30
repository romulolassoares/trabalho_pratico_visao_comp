import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  
import streamlit as st
from PIL import Image

from classes.AlexNet import AlexNet
from utils.functions import plot_confusion_matrix, load_cifar10

# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 32
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 100  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Defina as transformações para redimensionar e normalizar as imagens
transform = transforms.Compose([transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset, testset = load_cifar10(n_size=50, transform=transform)


# Defina o dataloader
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)




# Instancie a AlexNet
alexnet = AlexNet().to(device)


# Defina a função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9)
# Treine a rede
num_epochs = 2
train_losses = []
test_losses = []




for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    tqdm_trainloader = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}, Training', dynamic_ncols=True)

    for i, data in enumerate(tqdm_trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Mova os dados para o dispositivo correto

        optimizer.zero_grad()

        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tqdm_trainloader.set_postfix({'loss': running_loss / (i + 1)})

    # Calcule a perda média no conjunto de treinamento
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)

    # Avalie o modelo no conjunto de teste
    alexnet.eval()
    running_test_loss = 0.0
    tqdm_testloader = tqdm(testloader, desc=f'Epoch {epoch+1}/{num_epochs}, Testing', dynamic_ncols=True)

    with torch.no_grad():
        for data in tqdm_testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Mova os dados para o dispositivo correto

            outputs = alexnet(inputs)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

    # Calcule a perda média no conjunto de teste
    test_loss = running_test_loss / len(testloader)
    test_losses.append(test_loss)

    tqdm_trainloader.set_postfix({'Training Loss': train_loss, 'Testing Loss': test_loss})
    tqdm_trainloader.close()
    tqdm_testloader.close()

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plote gráficos de perda
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.show()


# Function to evaluate the model and calculate metrics
def evaluate_model(model, dataloader, classes):
    model.eval()
    model = model.to(device)  # Move the model to the correct device
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        tqdm_dataloader = tqdm(dataloader, desc='Evaluating Model', dynamic_ncols=True)
        for inputs, labels in tqdm_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move the data to the correct device
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Adicione o código para exibir as imagens aqui
            # Certifique-se de que os dados estejam no formato certo para serem exibidos pelo Streamlit

        tqdm_dataloader.close()

    # Calculate and print classification report
    st.write("Classification Report:")
    st.text(classification_report(all_labels, all_predictions, target_names=classes))

    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

# ... (o restante do seu código permanece inalterado)

# Use o Streamlit para exibir as imagens
st.title("CIFAR-10 Image Classification with AlexNet")

# Adicione o código Streamlit aqui para carregar e exibir as imagens

# Evaluate the model on the training set
st.subheader("Evaluation on Training Set")
evaluate_model(alexnet, trainloader, CIFAR10_CLASSES)

# Evaluate the model on the test set
st.subheader("Evaluation on Test Set")
evaluate_model(alexnet, testloader, CIFAR10_CLASSES)