import MultiClassificationTrainer as mct
from Dataset import Poker
from catboost import CatBoostClassifier

# Importing dataset
dataset = Poker([0.2, 0.1, 0.7], 0.02, None, False)

# Setting parameters, can be multiple
parameters = {
    'task_type': ['GPU'],
    'thread_count': [8],
    'classes_count': [10],
    'num_trees': [4000], # 1000, The maximum number of trees that can be built when solving machine learning problems.
    'learning_rate': [0.5], # 0.03, The learning rate. Used for reducing the gradient step.
    'max_depth': [10], # 6, Depth of the tree.
    'l2_leaf_reg': [1], # 3, L2 regularization coefficient. Used for leaf value calculation.
    'use_best_model': [True],
    'objective': ['MultiClassOneVsAll'], # RMSE, The metric to use in training. The specified value also determines the machine learning problem to solve.
    'eval_metric': ['TotalF1'] # Objective, The metric used for overfitting detection (if enabled) and best model selection (if enabled). 
    }

# Training model
model, parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, CatBoostClassifier, parameters, True)

# Analytics
print('Analyzing...')
metric_results = model.get_evals_result()['validation_0']
title = "CatBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean)
mct.analyze_and_save(title, dataset, y_pred, metric_results, parameters)