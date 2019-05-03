from lightning.classification import LinearSVC
from Evaluation import Evaluator
from Dataset import Poker
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
import time

# Create custom metric
gmean_scorer = make_scorer(score_func=geometric_mean_score, greater_is_better=True)

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "4SMOTE",
    #'sampling_strategy': "WSMOTE2",
    'sampling_strategy': None,
    'verbose': False
    }

dataset = Poker(**dataset_parameters)

model_parameters = {
    'verbose':1,
    'random_state':42,
    'max_iter':10,
    'C':3.0,
    'loss':'hinge',
    'criterion':'accuracy',
    'tol':0.001,
    'permute':True,
    'shrinking':True,
    'warm_start':False,
    'callback':None,
    'n_calls':100
    }

model = LinearSVC(**model_parameters)

# Training
print('Training...')
start_time = time.time()
model.fit(
    X=dataset.X_train.values,
    y=dataset.y_train.values)

elapsed_time_training = time.time() - start_time

# Predicting
print('Predicting...')
start_time = time.time()
y_pred = model.predict(dataset.X_test.values)
elapsed_time_testing = time.time() - start_time

# Analytics
#eval_results = {
#    'multi_logloss': model.evals_result_['valid_0']['multi_logloss'],
#    'gmean': np.absolute(model.evals_result_['valid_0']['gmean'])}

title = "ThunderSVM (nothing)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
#evaluator.append_to_file(f'Best iteration: {model.best_iteration_}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
#evaluator.append_to_file(eval_results, "metric_results.txt")
#evaluator.create_evaluation_metric_results(eval_results, xlabel='number of trees', ylabel='geometric mean')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()