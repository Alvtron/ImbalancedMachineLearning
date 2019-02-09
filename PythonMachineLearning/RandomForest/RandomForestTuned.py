import math
import numpy as np
import pandas as pd
from Performance import plot_confusion_matrix
from Dataset import create_poker_dataset
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib

predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights = create_poker_dataset()

# setting parameters
print('setting tuning parameters...')
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
    }

print('Tuning parameters:')
pprint(random_grid)

# Creating tree
print('Creating tree...')
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
classifier = RandomizedSearchCV(
    estimator = rf,
    param_distributions = random_grid,
    n_iter = 50,
    cv = 2,
    verbose=2,
    random_state=42,
    n_jobs = -1)

# training
print('Training...')
classifier.fit(X_train, y_train)

pprint(classifier.best_params_)