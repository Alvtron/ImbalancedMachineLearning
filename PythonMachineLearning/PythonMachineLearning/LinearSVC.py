from sklearn.svm import LinearSVC
from Evaluation import Evaluator
from Dataset import Poker
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
import time

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "over_and_under_sampling_custom",
    #'sampling_strategy': "UWSMOTE",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'verbose':2,
    'random_state':42,
    'penalty':'l2', # string, ‘l1’ or ‘l2’ (default=’l2’) Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
    'loss': 'squared_hinge', # string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’) Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
    'C':1.0, # float, optional (default=1.0) Penalty parameter C of the error term.
    'dual':True, # bool, (default=True) Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
    'class_weight':dataset.weight_per_class,
    'fit_intercept':False, # boolean, optional (default=True) Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (i.e. data is expected to be already centered).
    'intercept_scaling':1, # float, optional (default=1) When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
    'max_iter':10000, # int, (default=1000) The maximum number of iterations to be run
    'multi_class':'ovr', # string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’) Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.
    'tol':0.00001 # float, optional (default=1e-4) Tolerance for stopping criteria.
    }

model = LinearSVC(**model_parameters)

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

title = "LinearSVC (longrun minmax)"
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