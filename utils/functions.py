import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score,classification_report
from datetime import datetime

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

# ========================================================================================
# Evaluate Model
# ========================================================================================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    

def make_train(model, num_epochs, optimizer, criterion, train_loader, test_loader,device):
    metrics = []
    total_steps = 0
    for epoch in range(num_epochs):
        predicted_labels = []
        true_labels = []
        model.train()
        running_loss = 0.0
        
        # Treinamento
        tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}, Training', dynamic_ncols=True)
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
        # Calcule a perda média no conjunto de treinamento
        train_loss = running_loss / len(train_loader)

        # Teste
        model.eval()
        running_test_loss = 0.0
        tqdm_test_loader = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs}, Testing', dynamic_ncols=True)
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
        # Calcule a perda média no conjunto de teste
        test_loss = running_test_loss / len(test_loader)
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        tqdm_train_loader.set_postfix({'Training Loss': train_loss, 'Testing Loss': test_loss})
        tqdm_train_loader.close()
        tqdm_test_loader.close()
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        result_data = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        }
        metrics.append(result_data)
        
        total_steps+=1

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    return model, metrics




# Function to evaluate the model and calculate metrics
def evaluate_model(model, dataloader, classes,device):
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

    tqdm_dataloader.close()

    # Calculate and print classification report
    # print("Classification Report:")
    report_dict = classification_report(all_labels, all_predictions, target_names=classes,output_dict=True)

    # Calcular e plotar a matriz de confusão
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)

    # Configurar rótulos e título do gráfico, se desejar
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusão')

    # Retornar a figura e o objeto de eixo
    return fig, ax, report_dict
