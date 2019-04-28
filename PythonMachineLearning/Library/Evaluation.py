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
import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Evaluator:
    def __init__(self, title, save_path):
        now = time.time()
        self.title = title
        self.save_directory_path = save_path
        self.folder_path = f"{save_path}/{now} {title}/"
        os.makedirs(os.path.dirname(self.folder_path), exist_ok=True)

    def create_confusion_matrix(self, observed, prediction, classes, normalize=False, cmap=plt.cm.Blues):
        """
        This function plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
        """

        cnf_matrix = sklearn.metrics.confusion_matrix(observed, prediction, labels = classes)

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        plt.figure()
        plt.imshow(cnf_matrix, interpolation = 'nearest', cmap = cmap)
        #plt.title(f'{self.title} - Confusion matrix')
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

        plt.ylabel('Observed class')
        plt.xlabel('Predicted class')
        plt.tight_layout()
        plt.savefig(fname = f'{self.folder_path}confusion_matrix.png', format = 'png', dpi = 300)
        plt.savefig(fname = f'{self.folder_path}confusion_matrix.pdf', format = 'pdf')

    def create_evaluation_metric_results(self, metric_results, xlabel = 'iteration', ylabel = 'score'):
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.title(f'{self.title} - Evaluation metrics')

        if (isinstance(metric_results, dict)):
            for type, result in flatten(metric_results).items():
                line, = plt.plot(result, label=f"{type}")
                plt.legend()
        else:
            plt.plot(metric_results)
        plt.savefig(fname = f'{self.folder_path}evaluation_metrics.png', format = 'png', dpi = 300)
        plt.savefig(fname = f'{self.folder_path}evaluation_metrics.pdf', format = 'pdf')

    def save_advanced_metrics(self, observed, prediction, class_labels, class_descriptions):
        accuracy = sklearn.metrics.accuracy_score(observed, prediction)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(observed, prediction)
        sensitivity = imblearn.metrics.sensitivity_score(observed, prediction, labels = class_labels, average = 'macro')
        specificity = imblearn.metrics.specificity_score(observed, prediction, labels = class_labels, average = 'macro')
        geometric_mean = imblearn.metrics.geometric_mean_score(observed, prediction, labels = class_labels, average = 'macro')
        report = sklearn.metrics.classification_report(observed, prediction, labels = class_labels, target_names = class_descriptions)
        result = f"Accuracy: {accuracy}\nBalanced Accuracy: {balanced_accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nGeometric Mean: {geometric_mean}\n{report}"
        print(result)
        with open(f"{self.folder_path}score.txt", "w") as text_file:
            text_file.write(result)

    def append_to_file(self, object, file_name):
        with open(f"{self.folder_path}{file_name}", "a+") as text_file:
            if (isinstance(object, str)):
                text_file.write(f'{object}\n')
            elif (isinstance(object, dict)):
                for key, value in object.items():
                    if (isinstance(value, list)):
                        list_string = ",".join(map(str, value))
                        text_file.write(f'{key}: [{list_string}]\n')
                    else:
                        text_file.write(f'{key}: {value}\n')
            elif (isinstance(object, list)):
                for value in object:
                    text_file.write(f'{value}\n')
