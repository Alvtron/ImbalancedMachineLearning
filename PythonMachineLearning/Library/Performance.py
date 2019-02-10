import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def plot_confusion_matrix(observed, prediction, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """

    cnf_matrix = confusion_matrix(observed, prediction, labels = classes)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_evaluation_metric_results(metric_results, title = "Evaluation metric results"):
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Test')
    plt.title(title)
    plt.plot(metric_results)

def print_advanced_metrics(prediction, observed, class_descriptions):
    accuracy = accuracy_score(prediction, observed)
    print(f'Accuracy: {accuracy}')
    print('Advanced metrics:')
    print(classification_report(observed, prediction, target_names=class_descriptions))