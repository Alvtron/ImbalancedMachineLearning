import Performance as performance
import Dataset as dataset
import numpy as np
import time
from pprint import pprint
from itertools import product
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier

# Importing dataset
poker = dataset.Poker([0.2, 0.1, 0.7], 0.02, None, False)

parameters = {
    'task_type': 'GPU',
    'thread_count': 8,
    'classes_count': 10,
    'num_trees': 4000, # 1000, The maximum number of trees that can be built when solving machine learning problems.
    'learning_rate': 0.9, # 0.03, The learning rate. Used for reducing the gradient step.
    'max_depth': 10, # 6, Depth of the tree.
    'l2_leaf_reg': 1, # 3, L2 regularization coefficient. Used for leaf value calculation.
    'use_best_model': True,
    'objective': 'MultiClassOneVsAll', # RMSE, The metric to use in training. The specified value also determines the machine learning problem to solve.
    'eval_metric': 'TotalF1'} # Objective, The metric used for overfitting detection (if enabled) and best model selection (if enabled). 

# Creating model
print('Initializing model...')
model = CatBoostClassifier(**parameters)

# Training
print('Training...')
model.fit(
    X = poker.X_train,
    y = poker.y_train,
    sample_weight = poker.train_sample_weights,
    eval_set = [(poker.X_validate, poker.y_validate)],
    verbose = True)

# Predicting
print('Predicting...')
y_pred = model.predict(poker.X_test, prediction_type='Class')

# Analytics
metric_results = model.get_evals_result()['validation_0']
title = "CatBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(poker.sample_size * 100, poker.training_size * 100, poker.validation_size * 100, poker.testing_size * 100)
performance.write_parameters_to_file(title, parameters)
performance.plot_evaluation_metric_results(metric_results, f'{title} - Evaluation metrics')
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = f'{title} - Confusion matrix', normalize = True)
plt.show()