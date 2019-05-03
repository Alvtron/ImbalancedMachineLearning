from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
on the original dataset and then fits additional copies of the classifier
on the same dataset but where the weights of incorrectly classified instances
are adjusted such that subsequent classifiers focus more on difficult cases.

This class implements the algorithm known as AdaBoost-SAMME.

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

classifier_parameters = {
    'random_state': 42,
    'criterion': "gini", # The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    'splitter':"best", # The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    'max_depth': 10, # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    'min_samples_split': 2, # The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
    'min_samples_leaf': 1, # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
    'min_weight_fraction_leaf': 0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    'max_features': None, # The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split. If “auto”, then max_features=sqrt(n_features). If “sqrt”, then max_features=sqrt(n_features). If “log2”, then max_features=log2(n_features). If None, then max_features=n_features
    'max_leaf_nodes': None, # Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    'min_impurity_decrease': 0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    'class_weight': dataset.weight_per_class # Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    }

model_parameters = {
    'random_state': 42, # None, If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    'base_estimator': DecisionTreeClassifier(**classifier_parameters), # classifier, # The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier(max_depth=1)
    'algorithm': 'SAMME.R', # SAMME.R, If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
    'n_estimators': 500, # 50, The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
    'learning_rate': 0.5, # 1.0, Learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learning_rate and n_estimators.
    }

# Creating model
model = AdaBoostClassifier(**model_parameters)

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
title = "AdaBoost (hyper weight minmax)"
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
evaluator.create_evaluation_metric_results(early.scores_, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()