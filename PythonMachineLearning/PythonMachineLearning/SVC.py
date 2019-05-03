from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from Evaluation import Evaluator
from Dataset import Poker
from matplotlib import pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE2",
    #'sampling_strategy': "UWSMOTE",
    #'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'random_state': 42,
    'verbose': True,
    'cache_size': 4096, # float, optional. Specify the size of the kernel cache (in MB).
    'kernel': 'linear', # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.
    'class_weight': dataset.weight_per_class,
    'C':100.0, # float, optional (default=1.0). Penalty parameter C of the error term.
    'degree':3, # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    'gamma': 'auto', # float, optional. (default=’auto’) Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    'max_iter': -1, # int, optional (default=-1). Hard limit on iterations within solver, or -1 for no limit.
    'tol': 0.001, # float, optional. (default=1e-3) Tolerance for stopping criterion.
    'coef0': 0.0, # float, optional. (default=0.0) Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'decision_function_shape': 'ovr', # ‘ovo’, ‘ovr’, default=’ovr’. Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy.
    'probability': False, # boolean, optional. (default=False) Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
    'shrinking': True # boolean, optional. (default=True) Whether to use the shrinking heuristic.
    }

model = SVC(**model_parameters)

#model = BaggingClassifier(
#    verbose = 1,
#    base_estimator=SVC(**model_parameters),
#    n_estimators=8,
#    max_samples = 1 / 8,
#    bootstrap=False,
#    n_jobs=-1,
#    random_state=42)

# Training
print('Training...')
start_time = time.time()

model.fit(
    X = dataset.X_train,
    y = dataset.y_train)

elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

# Analytics
title = "SVC (hyper weights WSMOTE Bagging)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
#evaluator.append_to_file(eval_results, "metric_results.txt")
#evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()