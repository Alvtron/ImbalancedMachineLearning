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
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE",
    'sampling_strategy': None,
    'verbose': False
    }

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

model = thundersvm.SVC(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X=dataset.X_train.values,
    y=dataset.y_train.values)
#scores = cross_validate(
#    estimator=model,
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
y_pred = model.predict(dataset.X_test.values)
elapsed_time_testing = time.time() - start_time

# Analytics
#eval_results = {
#    'multi_logloss': model.evals_result_['valid_0']['multi_logloss'],
#    'gmean': np.absolute(model.evals_result_['valid_0']['gmean'])}

title = "ThunderSVM (test)"
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
evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()