import math
import numpy as np
import pandas as pd
from Performance import plot_confusion_matrix
from Dataset import create_poker_dataset
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights = create_poker_dataset()

# Creating tree
print('Creating tree...')
classifier = RandomForestClassifier(
    verbose = 2,
    n_jobs = -1,
    random_state = 42,
    criterion = 'entropy',
    n_estimators = 1800,
    max_depth = 30,
    bootstrap = False,
    max_features = 'sqrt',
    min_samples_leaf = 1,
    min_samples_split = 10)

# training
print('Training...')
classifier.fit(X_train, y_train, sample_weight = train_sample_weights)

# Saving model
print('Saving model...')
joblib.dump(classifier, 'randomforestmodel.pkl')

# Predicting
print('Predicting...')
y_pred = classifier.predict(X_test)

# Measuring accuracy
print('Accuracy:')
accuracy=accuracy_score(y_pred, y_test)
print(accuracy)

print('Advanced metrics:')
print(classification_report(y_test, y_pred, target_names=class_descriptions))

# Plotting confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_labels, title = 'Confusion matrix, with normalization', normalize = True)
plt.show()