from Evaluation import Evaluator
from Dataset import Poker
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import svm
from sklearn.model_selection import cross_validate, cross_val_score, PredefinedSplit

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE", # no significant improvement
    #'sampling_strategy': "over_and_under_sampling", # 10k and 20k shows promising for the first 8 classes, and 30-60% for class 9, but no hits on last class.
    #'sampling_strategy': "over_and_under_sampling_custom", # best result. 70% and 0% on two last classes, respectively.
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'C':1.0,
    'class_weight':dataset.weight_per_class,
    'dual':True,
    'intercept_scaling':1,
    'loss':'squared_hinge',
    'max_iter':1000,
    'multi_class':'ovr',
    'penalty':'l2',
    'random_state':42,
    'tol':0.0001,
    'verbose':0}

clf = svm.LinearSVC(**model_parameters)

# Training
print("Training...")
scores = cross_validate(
    estimator=clf,
    X=np.concatenate((dataset.X_train.values, dataset.X_validate.values)),
    y=np.concatenate((dataset.y_train.values, dataset.y_validate.values)),
    cv=PredefinedSplit(test_fold=[-1] * len(dataset.X_train.values) + [0] * len(dataset.X_validate.values)),
    scoring=['precision_macro', 'recall_macro'],
    return_train_score=False,
    n_jobs=-1,
    verbose=False)

# Predict
print("Predicting...")
predictions = saved_model.predict(dataset.X_test.values)
y_pred = []

for prediction in predictions:
    y_pred.append(np.argmax(prediction))

# Analytics
metric_results = model.history.history,
title = "Linear SVC ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100)

print('Analyzing...')
evaluator = Evaluator(title)
evaluator.write_model_parameters_to_file(model_parameters)
evaluator.write_dataset_parameters_to_file(dataset_parameters)
evaluator.plot_evaluation_metric_results(scores)
evaluator.plot_confusion_matrix(y_pred, dataset.y_test, dataset.class_labels, normalize = True)
evaluator.print_advanced_metrics(y_pred, dataset.y_test, dataset.class_labels, dataset.class_descriptions)
plt.show()