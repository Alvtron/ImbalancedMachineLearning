from Evaluation import Evaluator
from Dataset import Poker
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, cross_val_score, PredefinedSplit
import thundersvmScikit as thundersvm
import time

# Create custom metric
gmean_scorer = make_scorer(score_func=geometric_mean_score, greater_is_better=True)

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE", # no significant improvement
    #'sampling_strategy': "over_and_under_sampling", # 10k and 20k shows promising for the first 8 classes, and 30-60% for class 9, but no hits on last class.
    #'sampling_strategy': "over_and_under_sampling_custom", # best result. 70% and 0% on two last classes, respectively.
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'C':1.0,
    'cache_size':8192,
    'class_weight':dataset.weight_per_class,
    'coef0':0.0,
    'decision_function_shape':'ovr',
    'degree':3,
    'gamma':0.5,
    'kernel':'linear',
    'max_iter':-1,
    'probability':False,
    'random_state':42,
    'shrinking':True,
    'tol':0.0001,
    'verbose':True}

clf = thundersvm.SVC(**model_parameters)

# Training
print('Training...')
start_time = time.time()
clf.fit(X=dataset.X_train.values, y=dataset.y_train.values)
#scores = cross_validate(
#    estimator=clf,
#    X=np.concatenate((dataset.X_train.values, dataset.X_validate.values)),
#    y=np.concatenate((dataset.y_train.values, dataset.y_validate.values)),
#    cv=PredefinedSplit(test_fold=[-1] * len(dataset.X_train.values) + [0] * len(dataset.X_validate.values)),
#    scoring=gmean_scorer,
#    return_train_score=False,
#    verbose=True,
#    n_jobs=-1)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = clf.predict(dataset.X_test.values)
elapsed_time_testing = time.time() - start_time

# Analytics
title = "ThunderSVM ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100)

print('Analyzing...')
evaluator = Evaluator(title)
evaluator.write_model_parameters_to_file(model_parameters)
evaluator.write_dataset_parameters_to_file(dataset_parameters)
#evaluator.plot_evaluation_metric_results(metric_results)
evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
plt.show()