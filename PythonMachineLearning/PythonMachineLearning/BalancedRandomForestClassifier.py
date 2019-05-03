from imblearn.ensemble import BalancedRandomForestClassifier
from Evaluation import Evaluator
from Dataset import Poker
from EarlyStopping import EarlyStopping
from matplotlib import pyplot as plt
import time

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
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters
model_parameters = {
    'n_jobs': -1, # None, The number of jobs to run in parallel for both fit and predict. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    'random_state': 42,
    'verbose': 0,
    'n_estimators': 100, #200 # 100, The number of trees in the forest.
    'criterion':'gini', # string, optional (default=”gini”) The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.
    'max_depth': 25, # None, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    'min_samples_leaf': 3, # 1, The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
    'min_samples_split': 2, # 2, The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
    'min_weight_fraction_leaf': 0, # 0, The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    'max_features': 'auto', # 'auto', The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split. If “auto”, then max_features=sqrt(n_features). If “sqrt”, then max_features=sqrt(n_features) (same as “auto”). If “log2”, then max_features=log2(n_features). If None, then max_features=n_features.
    'max_leaf_nodes': None, # None, Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    'min_impurity_decrease': 0, # 0, A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    'bootstrap': False, # True, Whether bootstrap samples are used when building trees
    'oob_score': False, # False, Whether to use out-of-bag samples to estimate the generalization accuracy.
    'class_weight': dataset.weight_per_class, # None, Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    'sampling_strategy':'auto',
    'replacement': False # bool, optional (default=False) Whether or not to sample randomly with replacement or not.
    }

# Creating model
model = BalancedRandomForestClassifier(**model_parameters)

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
title = "BalancedRandomForestClassifier (hyper)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {early.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(early.eval_results, "metric_results.txt")
evaluator.create_evaluation_metric_results(early.eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()