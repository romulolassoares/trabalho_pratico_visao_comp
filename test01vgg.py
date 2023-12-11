# ========================================================================================
# Imports
# ========================================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

from classes.Vgg import Vgg
from utils.functionsVGG import *

from datetime import *
import os
import json
from keras.applications.vgg16 import preprocess_input
import pandas as pd

# ========================================================================================
# Variaveis que precisam ser alteradas em cada teste
# ========================================================================================
# Número de épocas
NUM_EPOCHS = 25
# Tamanho do lote em cada interação
BATCH_SIZE = 32
# Parâmetro de convergência
MOMENTUM = 0.9 
# Taxa de decaimento
LR_DECAY = 0.0005 
# Taxa de aprendizado inicial
LR_INIT = 0.01 
# Dimensão das imagens
IMAGE_DIM = 32 
# Lista de classes do dataset
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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

# Importa e ajusta
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images
train_labels = train_labels
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)


# modelo e compilação
DECAY_STEPS = 20*NUM_EPOCHS # valor obtido por pesquisa na internet

model = Vgg(NUM_CLASSES).model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_INIT,
    decay_steps= DECAY_STEPS,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=SGD(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# treinamento
history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(test_images,test_labels))

# ========================================================================================
# Salvando os arquivos de resultado
# ========================================================================================
path = './test/results'
base_name = f'e{NUM_EPOCHS}b{BATCH_SIZE}class{NUM_CLASSES}'
# Adicionando a data e hora ao nome da paprecision
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder = f'{path}/{base_name}_{current_time}'

os.makedirs(folder, exist_ok=True)

# salva info.json 
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

with open(f'{folder}/info.json', 'w') as json_file:
    json.dump(info, json_file, indent=2)

# Testa o modelo e salva as figuras de acuracia e perda
train_loss = history.history['loss']
train_acc= history.history['accuracy']
test_loss = history.history['val_loss']
test_acc = history.history['val_accuracy']
plot_loss_history(train_acc , train_loss, test_acc ,test_loss ,folder)

# Obtem report final 
pred_labels = model.predict(test_images)
metrics = metric_report(test_labels, pred_labels, folder)

df = pd.DataFrame({'Labels': CIFAR10_CLASSES,'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1-Score': metrics['f1']})
df.set_index('Labels')
df.to_csv(f'{folder}/report.csv',index=False)


