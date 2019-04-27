from Evaluation import Evaluator
from lightgbm import LGBMClassifier
from Dataset import Poker
import matplotlib.pyplot as plt
import time

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE", # 
    #'sampling_strategy': "over_and_under_sampling", # 
    #'sampling_strategy': "4SMOTE", #
    'sampling_strategy': "WSMOTE", # 
    #'sampling_strategy': None, #
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters
model_parameters = {
    #'boosting_type': 'gbdt', # ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
    'n_jobs': -1,
    'num_class': 10,
    #'n_estimators': 800, # 100, Number of boosted trees to fit.
    #'num_leaves': 60, # 31, Maximum tree leaves for base learners.
    #'max_depth': 10, # -1, Maximum tree depth for base learners, -1 means no limit.
    #'learning_rate': 0.4, # 0.1, Boosting learning rate.
    #'min_split_gain': 0, # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree.
    #'min_child_samples': 5, # 20, Minimum number of data needed in a child (leaf).
    #'subsample': 1, # 1, Subsample ratio of the training instance.
    #'subsample_freq': 0, # 0, Frequence of subsample, <=0 means no enable.
    #'colsample_bytree': 1, # 1, Subsample ratio of columns when constructing each tree.
    #'reg_alpha': 0, # 0, L1 regularization term on weights.
    #'reg_lambda': 0, # 0, L2 regularization term on weights.
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    #'class_weight': dataset.weight_per_class
    }

# Creating model
model = LGBMClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    eval_set = [(dataset.X_validate, dataset.y_validate)],
    verbose = True)
elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test, prediction_type='Class')
elapsed_time_testing = time.time() - start_time

# Analytics
metric_results = model.evals_result_
title = "LightGBM"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
print(f'Training time (seconds): {elapsed_time_training}')
print(f'Testing time (seconds): {elapsed_time_testing}')
evaluator = Evaluator(title, save_path)
evaluator.write_model_parameters_to_file(model_parameters)
evaluator.write_dataset_parameters_to_file(dataset_parameters)
evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
evaluator.plot_evaluation_metric_results(metric_results)
plt.show()