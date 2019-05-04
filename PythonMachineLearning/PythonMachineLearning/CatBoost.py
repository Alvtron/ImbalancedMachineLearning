from Evaluation import Evaluator
from Dataset import Poker
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score

class GeometricMean(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(target) == len(approxes[0])
        y_true = np.array(target)
        y_pred = np.array(approxes)
        y_pred = y_pred.argmax(axis = 0)
        return geometric_mean_score(y_true, y_pred, average = 'macro'), 0.0

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE2",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters, can be multiple
model_parameters = {
    'task_type': 'GPU',
    'thread_count': 8,
    'classes_count': 10,
    'num_trees': 100000, # 1000, The maximum number of trees that can be built when solving machine learning problems.
    'learning_rate': 0.10, # 0.03, The learning rate. Used for reducing the gradient step.
    'max_depth': 10, # 6, Depth of the tree.
    'l2_leaf_reg': 1, # 3, L2 regularization coefficient. Used for leaf value calculation.
    'objective': 'MultiClass', # MultiClassOneVsAll, RMSE, The metric to use in training. The specified value also determines the machine learning problem to solve.
    'eval_metric': 'MultiClass', #GeometricMean(), #"TotalF1", # Objective, The metric used for overfitting detection (if enabled) and best model selection (if enabled). 
    'custom_metric':'Accuracy',
    'od_type': 'Iter',
    'od_wait': 100,
    }

# Creating model
model = CatBoostClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    sample_weight = dataset.weight_per_sample,
    eval_set = [(dataset.X_validate, dataset.y_validate)],
    use_best_model=True,
    verbose = True)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test, prediction_type='Class')
elapsed_time_testing = time.time() - start_time

# Analytics
print('Analyzing...')
title = "CatBoost (weights smote)"

eval_results = {
    'MultiClass': np.absolute(model.get_evals_result()['validation_0']['MultiClass']),
    'Accuracy': np.absolute(model.get_evals_result()['validation_0']['Accuracy']),
    #'F1': np.absolute(model.get_evals_result()['validation_0']['TotalF1']),
    #'gmean': model.get_evals_result()['validation_0']['GeometricMean']
    }

save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {model.get_best_iteration()}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.save_dict_to_file(dataset_parameters, "dataset_parameters.csv")
evaluator.save_dict_to_file(model_parameters, "model_parameters.csv")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.save_eval_scores_to_file(eval_results, "metric_results.csv")
evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()