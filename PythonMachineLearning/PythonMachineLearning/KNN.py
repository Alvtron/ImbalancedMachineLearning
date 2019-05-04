from sklearn.neighbors import KNeighborsClassifier
from Evaluation import Evaluator
from Dataset import Poker
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
import time

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'min_max_scaling': True,
    #'sampling_strategy': "SMOTE",
    #'sampling_strategy': "over_and_under_sampling",
    #'sampling_strategy': "over_and_under_sampling_custom",
    #'sampling_strategy': "UWSMOTE",
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'n_jobs': -1,
    'n_neighbors': 5, # int, optional (default = 5) Number of neighbors to use by default for kneighbors queries.
    'weights': 'uniform', # str or callable, optional (default = ‘uniform') weight function used in prediction. Possible values: ‘uniform' : uniform weights. All points in each neighborhood are weighted equally. ‘distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away. [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
    'algorithm': 'auto', # {‘auto', ‘ball_tree', ‘kd_tree', ‘brute'}, optional Algorithm used to compute the nearest neighbors: ‘ball_tree' will use BallTree ‘kd_tree' will use KDTree ‘brute' will use a brute-force search. ‘auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method. Note: fitting on sparse input will override the setting of this parameter, using brute force.
    'leaf_size': 30, # int, optional (default = 30) Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
    'p': 2, # integer, optional (default = 2) Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    'metric': 'minkowski', # string or callable, default ‘minkowski' the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.
    'metric_params': None # dict, optional (default = None) Additional keyword arguments for the metric function.
    }

model = KNeighborsClassifier(**model_parameters)

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

title = "KNN ()"
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