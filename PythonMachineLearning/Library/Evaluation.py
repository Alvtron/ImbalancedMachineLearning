import os
import errno
import numpy as np
import pandas as pd
import time
import itertools
import imblearn.metrics
import sklearn.metrics
from matplotlib import pyplot as plt
import json
from CollectionExtensions import flatten

class Evaluator:
    def __init__(self, title):
        now = time.time()
        self.title = title
        self.file_path = f"../Library/Results/{now} {title}/{title}"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def plot_confusion_matrix(self, observed, prediction, classes, normalize=False, cmap=plt.cm.Blues):
        """
        This function plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
        """

        cnf_matrix = sklearn.metrics.confusion_matrix(observed, prediction, labels = classes)

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure()
        plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = cmap)
        plt.title(f'{self.title} - Confusion matrix')
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
        plt.savefig(fname = f'{self.file_path} - Confusion matrix.png', format = 'png', dpi = 300)
        plt.savefig(fname = f'{self.file_path} - Confusion matrix.svg', format = 'svg')

    def plot_evaluation_metric_results(self, metric_results):
        plt.figure()
        plt.xlabel('n')
        plt.ylabel('Value')
        plt.title(f'{self.title} - Evaluation metrics')
        plt.axhline(y = 0, linewidth=0.5, color = 'k')
        for type, result in flatten(metric_results, sep='_').items():
            line, = plt.plot(result, label=f"{type}")
            plt.legend()
        plt.savefig(fname = f'{self.file_path} - Evaluation metrics.png', format = 'png', dpi = 300)
        plt.savefig(fname = f'{self.file_path} - Evaluation metrics.svg', format = 'svg')

    def print_advanced_metrics(self, prediction, observed, class_labels, class_descriptions):
        accuracy = sklearn.metrics.accuracy_score(prediction, observed)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(prediction, observed)
        sensitivity = imblearn.metrics.sensitivity_score(prediction, observed, labels = class_labels, average = 'macro')
        specificity = imblearn.metrics.specificity_score(prediction, observed, labels = class_labels, average = 'macro')
        geometric_mean = imblearn.metrics.geometric_mean_score(prediction, observed, labels = class_labels, average = 'macro')
        report = sklearn.metrics.classification_report(observed, prediction, target_names = class_descriptions)
        result = f"Accuracy: {accuracy}\nBalanced Accuracy: {balanced_accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nGeometric Mean: {geometric_mean}\n{report}"
        print(result)
        with open(f"{self.file_path} - Metrics.txt", "w") as text_file:
            text_file.write(result)

    def write_model_parameters_to_file(self, parameters):
        with open(f"{self.file_path} - Model parameters.txt", "a") as text_file:
            for key, value in parameters.items():
                if (isinstance(value, list)):
                    list_string = ",".join(map(str, value))
                    text_file.write(f'{key}: [{list_string}]\n')
                else:
                    text_file.write(f'{key}: {value}\n')

    def write_dataset_parameters_to_file(self, parameters):
        with open(f"{self.file_path} - Dataset parameters.txt", "a") as text_file:
            for key, value in parameters.items():
                if (isinstance(value, list)):
                    list_string = ",".join(map(str, value))
                    text_file.write(f'{key}: [{list_string}]\n')
                else:
                    text_file.write(f'{key}: {value}\n')