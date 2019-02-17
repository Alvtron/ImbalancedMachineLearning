import numpy as np
import time
import imblearn.metrics
import sklearn.metrics
from pprint import pprint
from itertools import product
from matplotlib import pyplot as plt
from Evaluation import Evaluator
from sklearn.ensemble import RandomForestClassifier
from Dataset import Poker

def multiple_parameter_training(dataset, classifier, parameters, verbose = False):
    param_combinations = []
    for parameter in product(*parameters.values()):
        param_combinations.append(dict(zip(parameters, parameter)));
    if (verbose):
        print(f'Finding best parameters with {len(param_combinations)} combinations of')
        pprint(parameters)
    iteration = 0
    best_gmean = None
    best_model = None
    best_parameters = None
    best_y_pred = None
    for params in param_combinations:
        # Incrementing iterator
        iteration = iteration + 1
        # Setting time
        start_time = time.time()
        # Creating model
        model = classifier(**params)
        # Training
        if (classifier is RandomForestClassifier):
            model.fit(
                X = dataset.X_train,
                y = dataset.y_train)
        else:
            model.fit(
                X = dataset.X_train,
                y = dataset.y_train,
                sample_weight = dataset.train_sample_weights,
                eval_set = [(dataset.X_validate, dataset.y_validate)],
                verbose = verbose)
        # Predicting
        y_pred = model.predict(dataset.X_test)
        # Analyze prediction
        elapsed_time = time.time() - start_time
        accuracy = sklearn.metrics.accuracy_score(y_pred, dataset.y_test)
        geometric_mean = imblearn.metrics.geometric_mean_score(y_pred, dataset.y_test, labels = dataset.class_labels, average = 'macro')
        # Print progress
        if (verbose):
            print(params)
            print("{0}: time: {1:0.2f}s accuracy: {2:0.15f} gmean: {3:0.15f}".format(iteration, elapsed_time, accuracy, geometric_mean))
        # Is the best model?
        if(best_gmean is None or geometric_mean > best_gmean):
            best_gmean = geometric_mean
            best_model = model
            best_parameters = params
            best_y_pred = y_pred
    # Return best result
    return best_model, best_parameters, best_y_pred, best_gmean

def single_parameter_training(dataset, classifier, parameters, verbose = False):
    start_time = time.time()
    # Creating model
    if (verbose): print('Initializing model...')
    model = classifier(**parameters)
    # Training
    if (verbose): print('Training...')
    if (classifier is RandomForestClassifier):
        model.fit(
            X = dataset.X_train,
            y = dataset.y_train,
            sample_weight = dataset.train_sample_weights)
    else:
        model.fit(
            X = dataset.X_train,
            y = dataset.y_train,
            sample_weight = dataset.train_sample_weights,
            eval_set = [(dataset.X_validate, dataset.y_validate)],
            verbose = verbose)
    # Predicting
    if (verbose): print('Predicting...')
    y_pred = model.predict(dataset.X_test, prediction_type='Class')
    if (verbose): print('Training completed after {0:0.2f} seconds.'.format(time.time() - start_time))
    # Return result
    return model, parameters, y_pred

def analyze_and_save(title, dataset, y_pred, metric_results = None, parameters = None):
    evaluator = Evaluator(title)
    if (parameters is not None):
        evaluator.write_parameters_to_file(parameters)
    if (metric_results is not None):
        evaluator.plot_evaluation_metric_results(metric_results)
    evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
    evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
    plt.show()