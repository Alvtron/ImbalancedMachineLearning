import MultiClassificationTrainer as mct
from lightgbm import LGBMClassifier
from Dataset import Poker

# Importing dataset
dataset = Poker([0.2, 0.1, 0.7], 0.02, None, False)

# Setting parameters
parameters = {
    'boosting_type': ['gbdt'], # ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
    'n_jobs': [-1],
    'num_class': [10],
    'n_estimators': [1000], # 100, Number of boosted trees to fit.
    'num_leaves': [31], # 31, Maximum tree leaves for base learners.
    'max_depth': [7], # -1, Maximum tree depth for base learners, -1 means no limit.
    'learning_rate': [0.8], # 0.1, Boosting learning rate.
    'min_split_gain': [0], # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'min_child_samples': [10], # 20, Minimum number of data needed in a child (leaf).
    'subsample': [1], # 1, Subsample ratio of the training instance.
    'subsample_freq': [0], # 0, Frequence of subsample, <=0 means no enable.
    'colsample_bytree': [1], # 1, Subsample ratio of columns when constructing each tree.
    'reg_alpha': [0], # 0, L1 regularization term on weights.
    'reg_lambda': [0], # 0, L2 regularization term on weights.
    'objective': ['multiclass'],
    'metric': ['multi_logloss']
    }

# Training model
model, parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, LGBMClassifier, parameters, True)

# Analytics
print('Analyzing...')
metric_results = model.evals_result_['valid_0']
title = "LightGBM ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean)
mct.analyze_and_save(title, dataset, y_pred, metric_results, parameters)