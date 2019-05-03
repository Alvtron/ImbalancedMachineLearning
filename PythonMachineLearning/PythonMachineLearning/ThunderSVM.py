from Evaluation import Evaluator
from Dataset import Poker
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import BaggingClassifier
import thundersvm
import time

# Create custom metric
gmean_scorer = make_scorer(score_func=geometric_mean_score, greater_is_better=True)

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE2",
    #'sampling_strategy': "UWSMOTE",
    #'sampling_strategy': None,
    #'verbose': False
    }

dataset = Poker(**dataset_parameters)

model_parameters = {
    'verbose': True, # bool(default=False) enable verbose output. Note that this setting takes advantage of a per-process runtime setting; if enabled, ThunderSVM may not work properly in a multithreaded context.
    'random_state': 42, # int, RandomState instance or None, optional (default=None), not supported yet The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    'max_iter': -1, # int, optional (default=-1) hard limit on the number of iterations within the solver, or -1 for no limit.
    'n_jobs': -1, # int, optional (default=-1) set the number of cpu cores to use, or -1 for maximum.
    'max_mem_size': -1, # int, optional (default=-1) set the maximum memory size (MB) that thundersvm uses, or -1 for no limit.
    'class_weight': dataset.weight_per_class,  #{dict}, optional(default=None) set the parameter C of class i to weight*C, for C-SVC
    'kernel': 'linear', # string, optional(default='rbf') set type of kernel function 'linear': u'*v 'polynomial': (gamma*u'*v + coef0)^degree 'rbf': exp(-gamma*|u-v|^2) 'sigmoid': tanh(gamma*u'*v + coef0) 'precomputed' -- precomputed kernel (kernel values in training_set_file)
    'degree': 3, # int, optional(default=3) set degree in kernel function
    'gamma': 'auto', # float, optional(default='auto') set gamma in kernel function (auto:1/num_features)
    'coef0': 0.0, # float, optional(default=0.0) set coef0 in kernel function
    'C': 100.0, # optional(default=1.0) set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
    'tol': 0.1, # float, optional(default=0.001) set tolerance of termination criterion (default 0.001)
    'probability': False, # boolean, optional(default=False) whether to train a SVC or SVR model for probability estimates, True or False
    'shrinking': False, # boolean, optional (default=False, not supported yet for True) whether to use the shrinking heuristic.
    'decision_function_shape': 'ovr' # ‘ovo’, default=’ovo’, not supported yet for 'ovr' only for classifier. Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
    }

model = thundersvm.SVC(**model_parameters)
#model = BaggingClassifier(verbose = 1,base_estimator=modeld, n_estimators=8, max_samples = 1 / 8, bootstrap=False, n_jobs=-1, random_state=42)
# Training
print('Training...')
start_time = time.time()
model.fit(
    X=dataset.X_train.values,
    y=dataset.y_train.values)

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

title = "ThunderSVM (weights)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
#evaluator.append_to_file(f'Best iteration: {model.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
#evaluator.append_to_file(eval_results, "metric_results.txt")
#evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()