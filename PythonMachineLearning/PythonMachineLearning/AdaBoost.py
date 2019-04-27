from Evaluation import Evaluator
from Dataset import Poker
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time

'''

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
on the original dataset and then fits additional copies of the classifier
on the same dataset but where the weights of incorrectly classified instances
are adjusted such that subsequent classifiers focus more on difficult cases.

This class implements the algorithm known as AdaBoost-SAMME.

'''

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters, can be multiple
model_parameters = {
    'base_estimator': DecisionTreeClassifier(max_depth = 10), # The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier(max_depth=1)
    'n_estimators': 500, # 50, The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
    'learning_rate': 0.5, # 1.0, Learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learning_rate and n_estimators.
    'algorithm': 'SAMME.R', # SAMME.R, If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
    'random_state': 42, # None, If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    }

# Creating model
model = AdaBoostClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    sample_weight = dataset.weight)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_prob = model.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

y_pred = y_prob.argmax(axis=-1)

# Analytics
title = "AdaBoost"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
print(f'Training time (seconds): {elapsed_time_training}')
print(f'Testing time (seconds): {elapsed_time_testing}')
evaluator = Evaluator(title, save_path)
evaluator.write_model_parameters_to_file(model_parameters)
evaluator.write_dataset_parameters_to_file(dataset_parameters)
evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
#evaluator.plot_evaluation_metric_results(metric_results)
plt.show()