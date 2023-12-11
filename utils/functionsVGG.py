from matplotlib.pyplot import * 
import tensorflow as tf
from tensorflow import keras
import numpy
import sklearn as sk
from sklearn.metrics import *

def plot_loss_history(train_acc,train_loss, test_acc, test_loss ,folder):
    ''' Args: train_history (tf.model) \n 
    validation_results (list)'''
    subplot(1,2,1)
    plot(train_loss, label='Training Loss')
    plot(test_loss, label='Test Loss')
    legend()
    ylabel('Loss')
    title('Training and Test Loss Over Time')
    subplot(1,2,2)
    plot(train_acc, label='Training Accuracy')
    plot(test_loss, label='Test Accuracy')
    legend()
    ylabel('Accuracy')
    xlabel('Epochs')
    title('Training and Test Accuracy Over Time')
    savefig(f'{folder}/loss_over_time.pdf')

def metric_report(labels, predictions, folder):
    """Generates conf. matrix and save report files"""

    true = numpy.argmax(labels, axis= 1)
    pred = numpy.argmax(predictions, axis =1)
    
    precision_list = precision_score(true, pred, average=None)
    recall_list = recall_score(true, pred, average= None)
    f1_list = f1_score(true, pred, average = None)
    
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                display_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    disp.plot()
    savefig(f'{folder}/confusion_matrix.pdf')

    return {'precision': precision_list, 'recall': recall_list,'f1': f1_list, 'matrix': disp}
