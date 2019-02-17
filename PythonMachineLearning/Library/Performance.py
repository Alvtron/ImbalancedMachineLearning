import numpy as np
import time
import itertools
import imblearn.metrics
import sklearn.metrics
from matplotlib import pyplot as plt

def plot_confusion_matrix(observed, prediction, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """

    cnf_matrix = sklearn.metrics.confusion_matrix(observed, prediction, labels = classes)

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
    now = time.time()
    plt.savefig(fname = f'../Library/Plots/{now}_{title}.png', format = 'png', dpi = 300)
    plt.savefig(fname = f'../Library/Plots/{now}_{title}.svg', format = 'svg')

def plot_evaluation_metric_results(metric_results, title = "Evaluation metric results"):
    plt.figure()
    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title(title)
    plt.axhline(y = 0, linewidth=0.5, color = 'k')
    plt.axhline(y = 1, linewidth=0.5, color = 'k')
    for metric_type, metric_result in metric_results.items():
        line, = plt.plot(metric_results[metric_type], label=metric_type)
        plt.legend()
    now = time.time()
    plt.savefig(fname = f'../Library/Plots/{now}_{title}.png', format = 'png', dpi = 300)
    plt.savefig(fname = f'../Library/Plots/{now}_{title}.svg', format = 'svg')

def print_advanced_metrics(prediction, observed, class_labels, class_descriptions):
    accuracy = sklearn.metrics.accuracy_score(prediction, observed)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(prediction, observed)
    sensitivity = imblearn.metrics.sensitivity_score(prediction, observed, labels = class_labels, average = 'macro')
    specificity = imblearn.metrics.specificity_score(prediction, observed, labels = class_labels, average = 'macro')
    geometric_mean = imblearn.metrics.geometric_mean_score(prediction, observed, labels = class_labels, average = 'macro')
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Geometric Mean: {geometric_mean}")
    print(sklearn.metrics.classification_report(observed, prediction, target_names=class_descriptions))

def write_parameters_to_file(title, parameters):
    now = time.time()
    with open(f"../Library/Plots/{now}_{title}_parameters.txt", "w") as text_file:
        text_file.write(parameters)