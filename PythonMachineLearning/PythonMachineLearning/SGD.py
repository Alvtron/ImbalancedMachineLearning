from sklearn.linear_model import SGDClassifier

from Evaluation import Evaluator
from Dataset import Poker
from EarlyStopping import EarlyStopping

import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from functools import partial

def scorer(X, y, classifier):
    prediction = classifier.predict(X)
    return { "gmean": geometric_mean_score(y, prediction, average = 'macro'), "accuracy":accuracy_score(y, prediction) }

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'verbose':1, # The verbosity level
    'max_iter':100,
    'n_jobs':-1, # The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation.
    'random_state':42, # The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator
    'class_weight':dataset.weight_per_class, # Preset for the class_weight fit parameter. Weights associated with classes. If not given, all classes are supposed to have weight one.
    'loss':'hinge', # The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.
    'penalty':'l2', # The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
    'alpha':0.0001, # 0.0001, Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.
    'l1_ratio':0.15, # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
    'fit_intercept':False, # Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.
    'average':False, # When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute. If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.
    'epsilon':0.1, # Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
    'learning_rate':'constant', # optimal, The learning rate schedule: 'constant': eta = eta0 'optimal': [default] eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou. 'invscaling': eta = eta0 / pow(t, power_t) 'adaptive': eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.
    'eta0':0.9, # The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
    'power_t':0.5, # The exponent for inverse scaling learning rate [default 0.5].
    'shuffle':False, # Whether or not the training data should be shuffled after each epoch. Defaults to True.
    'tol':None, # The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to None. Defaults to 1e-3 from 0.21.
    }

model = SGDClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()

model.fit(
    X = dataset.X_train,
    y = dataset.y_train)

#early = EarlyStopping(
#    model,
#    max_n_estimators=10000,
#    scorer=partial(scorer, dataset.X_validate, dataset.y_validate),
#    sample_weight=dataset.weights_per_class
#    monitor_score = "gmean",
#    patience = 500,
#    higher_is_better = True)

#early.fit(dataset.X_train, dataset.y_train)

elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
#y_pred = early.estimator.predict(dataset.X_test)
y_pred = model.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

# Analytics
title = "SGD (weights)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
#evaluator.append_to_file(f'Best iteration: {early.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
#evaluator.append_to_file(early.eval_results, "metric_results.txt")
#evaluator.create_evaluation_metric_results(early.eval_results, xlabel='epochs', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()