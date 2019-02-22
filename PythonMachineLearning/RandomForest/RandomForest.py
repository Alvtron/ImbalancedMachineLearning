import MultiClassificationTrainer as mct
from sklearn.ensemble import RandomForestClassifier
from Dataset import Poker

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': None}

dataset = Poker(**dataset_parameters)

# Setting parameters
model_parameters = {
    'n_jobs': [-1], # None, The number of jobs to run in parallel for both fit and predict. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    'n_estimators': [100, 200, 50], # 100, The number of trees in the forest.
    'random_state': [42],
    'verbose': [2],
    'max_depth': [10, 25], # None, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    'min_samples_leaf': [1,3], # 1, The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
    'min_samples_split': [2, 5], # 2, The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
    'min_weight_fraction_leaf': [0], # 0, The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    'max_features': ['auto'], # 'auto', The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split. If “auto”, then max_features=sqrt(n_features). If “sqrt”, then max_features=sqrt(n_features) (same as “auto”). If “log2”, then max_features=log2(n_features). If None, then max_features=n_features.
    'max_leaf_nodes': [None], # None, Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    'min_impurity_decrease': [0], # 0, A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    'bootstrap': [True, False], # True, Whether bootstrap samples are used when building trees
    'oob_score': [False], # False, Whether to use out-of-bag samples to estimate the generalization accuracy.
    'class_weight': [None], # None, Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    }

# Training model
model, model_parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, RandomForestClassifier, model_parameters, True)

# Analytics
print('Analyzing...')
mct.analyze_and_save(
    title = "RandomForest ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean),
    dataset = dataset,
    y_pred = y_pred,
    model_parameters = model_parameters,
    dataset_parameters = dataset_parameters)