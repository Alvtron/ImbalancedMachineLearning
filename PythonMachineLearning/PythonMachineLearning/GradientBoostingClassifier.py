from sklearn.ensemble import GradientBoostingClassifier
from Evaluation import Evaluator
from Dataset import Poker
from EarlyStopping import EarlyStopping
from matplotlib import pyplot as plt
import time

from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from functools import partial

'''

Gradient Boosting for classification.

GB builds an additive model in a forward stage-wise fashion;
it allows for the optimization of arbitrary differentiable loss functions.
In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function.
Binary classification is a special case where only a single regression tree is induced.

'''

def scorer(X, y, classifier):
    prediction = classifier.predict(X)
    return { "gmean": geometric_mean_score(y, prediction, average = 'macro'), "accuracy":accuracy_score(y, prediction) }

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE",
    'sampling_strategy': None,
    'verbose': False
    }

dataset = Poker(**dataset_parameters)

model_parameters = {
    'warm_start': True,
    'verbose': 1,
    'random_state': 42,
    'n_estimators': 1,
    'loss': 'deviance',
    'learning_rate': 0.7, # 0.1
    'subsample': 0.5, # 1.0
    'criterion': 'friedman_mse',
    'min_samples_split': 2, # 2
    'min_samples_leaf': 3, # 1
    'min_weight_fraction_leaf': 0.0,
    'max_depth': 5,
    'min_impurity_decrease': 0.0,
    'max_features': 'auto', # None
    'max_leaf_nodes': None,
    'presort': 'auto'
    }

# Creating model
model = GradientBoostingClassifier(**model_parameters)

# Training
print('Training...')

#model.fit(
#    X = dataset.X_train,
#    y = dataset.y_train)

early = EarlyStopping(
    model,
    max_n_estimators=1000,
    scorer=partial(scorer, dataset.X_validate, dataset.y_validate),
    monitor_score = "gmean",
    sample_weight=dataset.weight_per_sample,
    patience = 25,
    higher_is_better = True)

start_time = time.time()
early.fit(dataset.X_train, dataset.y_train)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = early.estimator.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

# Analytics
title = "GradientBoostingClassifier (hyper weights)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {early.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(classifier_parameters, "classifier_parameters.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(early.scores_, "metric_results.txt")
evaluator.create_evaluation_metric_results(early.scores_, xlabel='boosting stages', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()