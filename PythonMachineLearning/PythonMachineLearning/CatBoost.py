from Evaluation import Evaluator
from Dataset import Poker
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score

class LoggLoss(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers
        # (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.   
        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(target) == len(approxes[0])

        error_sum = 0.0
        weight_sum = 0.0

        for groups in approxes:
            for i, approx in enumerate(groups):
                w = 1.0 if weight is None else weight[i]
                weight_sum += w
                e = w * (target[i] * approx - math.log(1 + math.exp(approx)))
                error_sum += e

        return error_sum, weight_sum

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

# Setting parameters, can be multiple
model_parameters = {
    'task_type': 'GPU',
    'thread_count': 8,
    'classes_count': 10,
    'num_trees': 100000, # 1000, The maximum number of trees that can be built when solving machine learning problems.
    #'learning_rate': 0.55, # 0.03, The learning rate. Used for reducing the gradient step.
    #'max_depth': 10, # 6, Depth of the tree.
    #'l2_leaf_reg': 1, # 3, L2 regularization coefficient. Used for leaf value calculation.
    'objective': 'MultiClassOneVsAll', # RMSE, The metric to use in training. The specified value also determines the machine learning problem to solve.
    'eval_metric': "TotalF1", # Objective, The metric used for overfitting detection (if enabled) and best model selection (if enabled). 
    'od_type': 'Iter',
    'od_wait': 50,
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

eval_results = {
    'TotalF1': model.get_evals_result()['validation_0']['TotalF1'],
    'MultiClassOneVsAll': model.get_evals_result()['validation_0']['MultiClassOneVsAll']}

title = "CatBoost (weights)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {model.tree_count_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(eval_results, "metric_results.txt")
evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()