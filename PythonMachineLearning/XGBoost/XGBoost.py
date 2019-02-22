import MultiClassificationTrainer as mct
from xgboost import XGBClassifier
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
    'booster': ['gbtree'], # gbtree, Which booster to use: gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
    'nthread': [8], # Number of parallel threads used to run XGBoost
    'num_class': [10],
    'n_estimators': [250], # Number of boosted trees to fit.
    'learning_rate': [0.5], # 0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    'gamma': [0.5, 1.0, 2.0], # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    'max_depth': [10], # 6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    'min_child_weight': [1], # Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'max_delta_step': [0], # Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    'reg_aplha': [0], # L1 regularization term on weights. Increasing this value will make model more conservative.
    'reg_lambda': [0], # L2 regularization term on weights. Increasing this value will make model more conservative.
    'objective': ['multi:softmax'],
    'eval_metric': ['mlogloss']
    }

# Training model
model, model_parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, XGBClassifier, model_parameters, True)

# Analytics
print('Analyzing...')
mct.analyze_and_save(
    title = "XGBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean),
    dataset = dataset,
    y_pred = y_pred,
    metric_results = model.evals_result(),
    model_parameters = model_parameters,
    dataset_parameters = dataset_parameters)