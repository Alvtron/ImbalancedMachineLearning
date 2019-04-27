from Evaluation import Evaluator
from Dataset import Poker
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import time

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': None
    }

dataset = Poker(**dataset_parameters)

# Setting parameters, can be multiple
model_parameters = {
    'task_type': 'GPU',
    'thread_count': 8,
    'classes_count': 10,
    #num_trees': 2000, # 1000, The maximum number of trees that can be built when solving machine learning problems.
    #learning_rate': 0.55, # 0.03, The learning rate. Used for reducing the gradient step.
    #max_depth': 10, # 6, Depth of the tree.
    #l2_leaf_reg': 1, # 3, L2 regularization coefficient. Used for leaf value calculation.
    #use_best_model': True,
    'objective': 'MultiClassOneVsAll', # RMSE, The metric to use in training. The specified value also determines the machine learning problem to solve.
    'eval_metric': 'TotalF1' # Objective, The metric used for overfitting detection (if enabled) and best model selection (if enabled). 
    }

# Creating model
model = CatBoostClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    #sample_weight = dataset.weight_per_sample,
    eval_set = [(dataset.X_validate, dataset.y_validate)],
    verbose = True)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test, prediction_type='Class')
elapsed_time_testing = time.time() - start_time

# Analytics
metric_results = model.get_evals_result()
title = "CatBoost"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
print(f'Training time (seconds): {elapsed_time_training}')
print(f'Testing time (seconds): {elapsed_time_testing}')
evaluator = Evaluator(title, save_path)
evaluator.write_model_parameters_to_file(model_parameters)
evaluator.write_dataset_parameters_to_file(dataset_parameters)
evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
evaluator.plot_evaluation_metric_results(metric_results)
plt.show()