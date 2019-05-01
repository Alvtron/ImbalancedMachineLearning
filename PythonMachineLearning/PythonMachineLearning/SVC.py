from sklearn import svm
from Evaluation import Evaluator
from Dataset import Poker
from EarlyStopping import EarlyStopping
from matplotlib import pyplot as plt
import time

from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from functools import partial

def scorer(X, y, classifier):
    prediction = classifier.predict(X)
    return { "gmean": geometric_mean_score(y, prediction, average = 'macro'), "accuracy":accuracy_score(y, prediction) }

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'C':1.0,
    'cache_size':4096,
    'class_weight':dataset.weight_per_class,
    'coef0':0.0,
    'decision_function_shape':'ovr',
    'degree':3,
    'gamma':'scale',
    'kernel':'rbf',
    'max_iter':-1,
    'probability':False,
    'random_state':42,
    'shrinking':True,
    'tol':0.001,
    'verbose':True}

model = svm.SVC(**model_parameters)

# Training
print('Training...')

#model.fit(
#    X = dataset.X_train,
#    y = dataset.y_train)

early = EarlyStopping(
    model,
    max_n_estimators=1000,
    scorer=partial(scorer, dataset.X_validate, dataset.y_validate),
    monitor_score = "gmean",
    patience = 25,
    higher_is_better = True)

start_time = time.time()
early.fit(dataset.X_train, dataset.y_train)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = early.estimator.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

# Analytics
title = "RandomForest"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {early.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(early.scores_, "metric_results.txt")
evaluator.create_evaluation_metric_results(early.scores_, xlabel='number of trees', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()