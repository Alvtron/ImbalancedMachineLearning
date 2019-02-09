import math
import numpy as np
import pandas as pd
from Performance import plot_confusion_matrix
from Dataset import create_poker_dataset
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool

predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights = create_poker_dataset()

#print('Loading existing model...')
#load_model("catboost_model", format='catboost')

#model = CatBoostClassifier(
#    task_type = 'GPU',
#    thread_count = 8,
#    num_trees = 10000,
#    depth = 4,
#    learning_rate = 0.29,
#    loss_function = 'MultiClass',
#    boosting_type = 'Plain',
#    classes_count = 10,
#    logging_level = 'Verbose')

# 0.871506
print('Initializing model...')
model = CatBoostClassifier(
    task_type = 'GPU',
    thread_count = 8,
    num_trees = 1000,
    depth = 4,
    learning_rate = 0.29,
    loss_function = 'MultiClass',
    boosting_type = 'Plain',
    classes_count = 10,
    logging_level = 'Verbose')

print('Training...')
model.fit(X = X_train, y = y_train, sample_weight = train_sample_weights)

# save model
#print('Saving model...')
#model.save_model("catboost_model", format="cbm")

# make the prediction using the resulting model
print('Predicting...')
y_pred = model.predict(X_test, prediction_type='Class',)

# Measuring accuracy
print('Accuracy:')
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

print('Advanced metrics:')
print(classification_report(y_test, y_pred, target_names=class_descriptions))

# Plotting confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_labels, title = 'Confusion matrix, with normalization', normalize = True)
plt.show()