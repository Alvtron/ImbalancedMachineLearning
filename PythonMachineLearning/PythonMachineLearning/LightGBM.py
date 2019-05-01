from lightgbm import LGBMClassifier
from Evaluation import Evaluator
from Dataset import Poker
from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score

def accuracy_metric(preds, dtrain):
    predictions = y_pred.reshape(len(np.unique(y_true)), -1)
    predictions = predictions.argmax(axis = 0)
    return 'accuracy', geometric_mean_score(y_true, predictions), True

def gmean_metric(y_true, y_pred):
    predictions = y_pred.reshape(len(np.unique(y_true)), -1)
    predictions = predictions.argmax(axis = 0)
    return 'gmean', geometric_mean_score(y_true, predictions, average = 'macro'), True

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "WSMOTE2", 
    #'sampling_strategy': "WSMOTE4", 
    #'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters
model_parameters = {
    'boosting_type': 'gbdt', # ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
    'n_jobs': -1,
    'num_class': 10,
    'n_estimators': 100000, #800# 100, Number of boosted trees to fit.
    'class_weight': dataset.weight_per_class,
    'num_leaves': 60, # 31, Maximum tree leaves for base learners.
    'max_depth': 10, # -1, Maximum tree depth for base learners, -1 means no limit.
    'learning_rate': 0.4, # 0.1, Boosting learning rate.
    'min_split_gain': 0, # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'min_child_samples': 5, # 20, Minimum number of data needed in a child (leaf).
    'subsample': 1, # 1, Subsample ratio of the training instance.
    'subsample_freq': 0, # 0, Frequence of subsample, <=0 means no enable.
    'colsample_bytree': 1, # 1, Subsample ratio of columns when constructing each tree.
    'reg_alpha': 0, # 0, L1 regularization term on weights.
    'reg_lambda': 0, # 0, L2 regularization term on weights.
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    }

# Creating model
model = LGBMClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    early_stopping_rounds=50,
    eval_metric = gmean_metric,
    eval_set = [(dataset.X_validate, dataset.y_validate)],
    verbose = True)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test, prediction_type='Class')
elapsed_time_testing = time.time() - start_time

# Analytics
title = "LightGBM (hyper weights)"

eval_results = {
    'multi_logloss': model.evals_result_['valid_0']['multi_logloss'],
    'gmean': model.evals_result_['valid_0']['gmean']}

save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {model.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(eval_results, "metric_results.txt")
evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()