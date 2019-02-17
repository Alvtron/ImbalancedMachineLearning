import xgboost as xgb
from xgboost import plot_tree
from xgboost import XGBClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

# Importing dataset
poker = dataset.Poker([0.2, 0.1, 0.7], 0.02, None, False)

# Setting parameters
parameters= {
    'booster': 'gbtree', # gbtree, Which booster to use: gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
    'nthread': 8, # Number of parallel threads used to run XGBoost
    'num_class': 10,
    'n_estimators': 1000, # Number of boosted trees to fit.
    'learning_rate': 0.5, # 0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    'gamma': 0, # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    'max_depth': 6, # 6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    'min_child_weight': 1, # Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'max_delta_step': 0, # Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    'reg_lambda': 1, # L2 regularization term on weights. Increasing this value will make model more conservative.
    'reg_aplha': 0, # L1 regularization term on weights. Increasing this value will make model more conservative.
    'eval_metric': ['mlogloss', 'merror'],
    'objective': 'multi:softmax'
    }

# Creating model
print('Creating model with tuning parameters...')
model = XGBClassifier(**parameters)

# Training
print('Training...')
model.fit(
    X = poker.X_train,
    y = poker.y_train,
    sample_weight = poker.train_sample_weights,
    eval_set = [(poker.X_validate, poker.y_validate)],
    verbose = True)

# Predicting
print('Predicting...')
y_pred = model.predict(poker.X_test)

# Analytics
metric_results = model.evals_result()['validation_0']
title = "XGBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(poker.sample_size * 100, poker.training_size * 100, poker.validation_size * 100, poker.testing_size * 100)
performance.write_parameters_to_file(title, parameters)
performance.plot_evaluation_metric_results(metric_results, f'{title} - Evaluation metrics')
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = f'{title} - Confusion matrix', normalize = True)
plt.show()