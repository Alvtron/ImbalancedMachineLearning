import MultiClassificationTrainer as mct
from lightgbm import LGBMClassifier
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
    'boosting_type': ['gbdt'], # ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
    'n_jobs': [-1],
    'num_class': [10],
    'n_estimators': [800], # 100, Number of boosted trees to fit.
    'num_leaves': [60], # 31, Maximum tree leaves for base learners.
    'max_depth': [10], # -1, Maximum tree depth for base learners, -1 means no limit.
    'learning_rate': [0.4], # 0.1, Boosting learning rate.
    'min_split_gain': [0], # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'min_child_samples': [5], # 20, Minimum number of data needed in a child (leaf).
    'subsample': [1], # 1, Subsample ratio of the training instance.
    'subsample_freq': [0], # 0, Frequence of subsample, <=0 means no enable.
    'colsample_bytree': [1], # 1, Subsample ratio of columns when constructing each tree.
    'reg_alpha': [0], # 0, L1 regularization term on weights.
    'reg_lambda': [0], # 0, L2 regularization term on weights.
    'objective': ['multiclass'],
    'metric': ['multi_logloss']
    }

# Training model
model, model_parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, LGBMClassifier, model_parameters, True)

# Analytics
print('Analyzing...')
mct.analyze_and_save(
    title = "LightGBM ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean),
    dataset = dataset,
    y_pred = y_pred,
    metric_results = model.evals_result_,
    model_parameters = model_parameters,
    dataset_parameters = dataset_parameters)