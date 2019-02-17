from lightgbm import LGBMClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

# Importing dataset
poker = dataset.Poker([0.2, 0.1, 0.7], 0.02, None, False)

# Setting parameters
parameters = {
    'boosting_type': 'gbdt', # ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
    'num_class': 10,
    'n_estimators': 200, # 100, Number of boosted trees to fit.
    'num_leaves': 60, # 31, Maximum tree leaves for base learners.
    'max_depth': -1, # -1, Maximum tree depth for base learners, -1 means no limit.
    'learning_rate': 0.5, # 0.1, Boosting learning rate.
    'min_split_gain': 0, # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    'min_child_samples': 20, # 20, Minimum number of data needed in a child (leaf).
    'subsample': 1, # 1, Subsample ratio of the training instance.
    'subsample_freq': 0, # 0, Frequence of subsample, <=0 means no enable.
    'colsample_bytree': 1, # 1, Subsample ratio of columns when constructing each tree.
    'reg_alpha': 0, # 0, L1 regularization term on weights.
    'reg_lambda': 0, # 0, L2 regularization term on weights.
    'objective': 'multiclass',
    'metric': ['multi_logloss', 'multi_error']}

# Creating model
print('Initializing model...')
model = LGBMClassifier(**parameters)

# Training
print('Training...')
model.fit(X = poker.X_train, y = poker.y_train, sample_weight = poker.train_sample_weights, eval_set = [(poker.X_validate, poker.y_validate)], verbose = True)

# Predicting
print('Predicting...')
y_pred = model.predict(poker.X_test)

# Analytics
metric_results = model.evals_result_['valid_0']
title = "LightGBM ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(poker.sample_size * 100, poker.training_size * 100, poker.validation_size * 100, poker.testing_size * 100)
performance.write_parameters_to_file(title, parameters)
performance.plot_evaluation_metric_results(metric_results, f'{title} - Evaluation metrics')
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = f'{title} - Confusion matrix', normalize = True)
plt.show()