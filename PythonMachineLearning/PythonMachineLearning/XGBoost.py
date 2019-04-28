from Evaluation import Evaluator
import MultiClassificationTrainer as mct
from xgboost import XGBClassifier
from Dataset import Poker
from matplotlib import pyplot as plt
import numpy as np
import time

from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from functools import partial

def focal_binary_object(pred,dtrain,gamma_indct=1.2):
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    grad_first_part = (label+((-1)**label)*sigmoid_pred)**gamma_indct
    grad_second_part = label - sigmoid_pred
    grad_third_part = gamma_indct*(1-label-sigmoid_pred)
    grad_log_part = np.log(1-label-((-1)**label)*sigmoid_pred + 1e-7)       # add a small number to avoid numerical instability
    # combine the gradient
    grad = -grad_first_part*(grad_second_part+grad_third_part*grad_log_part)
    # combine the gradient parts to get hessian
    hess_first_term = gamma_indct*(label+((-1)**label)*sigmoid_pred)**(gamma_indct-1)*sigmoid_pred*(1.0 - sigmoid_pred)*(grad_second_part+grad_third_part*grad_log_part)
    hess_second_term = (-sigmoid_pred*(1.0 - sigmoid_pred)-gamma_indct*sigmoid_pred*(1.0 - sigmoid_pred)*grad_log_part-((1/(1-label-((-1)**label)*sigmoid_pred))*sigmoid_pred*(1.0 - sigmoid_pred)))*grad_first_part
    # get the final 2nd order derivative
    hess = -(hess_first_term+hess_second_term)
    
    return grad, hess

def accuracy_metric(preds, dtrain):
    return 'accuracy', accuracy_score(dtrain.get_label(), np.asarray([np.argmax(line) for line in preds]))

def gmean_metric(preds, dtrain):
    return 'gmean', -geometric_mean_score(dtrain.get_label(), np.asarray([np.argmax(line) for line in preds]), average = 'macro')

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    'sampling_strategy': "WSMOTE",
    #'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters
model_parameters = {
    'n_jobs': -1, # None, The number of jobs to run in parallel for both fit and predict. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    'random_state': 42,
    'verbosity': 0,
    'num_class': 10,
    'booster': 'gbtree', # gbtree, Which booster to use: gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
    'n_estimators': 1000, # Number of boosted trees to fit.
    'learning_rate': 0.5, # 0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    'gamma': 0.5, # 0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    'max_depth': 10, # 6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    'min_child_weight': 1, # Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'max_delta_step': 0, # Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    'reg_aplha': 0, # L1 regularization term on weights. Increasing this value will make model more conservative.
    'reg_lambda': 0, # L2 regularization term on weights. Increasing this value will make model more conservative.
    'objective': 'multi:softmax',
    'eval_metric':'mlogloss'
    }

# Creating model
model = XGBClassifier(**model_parameters)

start_time = time.time()

model.fit(
    X = dataset.X_train,
    y = dataset.y_train,
    sample_weight = dataset.weight_per_sample,
    early_stopping_rounds=25,
    eval_metric = gmean_metric,
    eval_set = [(dataset.X_validate, dataset.y_validate)],
    verbose = True)

elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test)
elapsed_time_testing = time.time() - start_time

# Analytics

eval_results = {
    'mlogloss': model.evals_result()['validation_0']['mlogloss'],
    'gmean': np.absolute(model.evals_result()['validation_0']['gmean'])}

title = "XGBoost (HYPER)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {model.best_iteration}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(eval_results, "metric_results.txt")
evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()