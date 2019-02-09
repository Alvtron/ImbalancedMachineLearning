import math
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree
from xgboost import XGBClassifier
from Performance import plot_confusion_matrix
from Dataset import create_poker_dataset
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights = create_poker_dataset()

# Setting parameters

# 91.2181 %
#params = {
#    'max_depth': 4,
#    'eta': 0.9,
#    'silent': True,
#    'objective': 'multi:softprob',
#    'num_class': 10}

# Training model
#bst = xgb.train(params, dtrain, num_round)

print('Creating model with tuning parameters...')
bst = XGBClassifier(
    nthread = 8,
    n_jobs = -1,
    num_class = 10,
    n_estimators= 1000,
    max_depth = 4,
    learning_rate = 0.5,
    eval_metric = 'mlogloss',
    objective = 'multi:softmax')

print('Training...')
bst.fit(
    X = X_train,
    y = y_train,
    sample_weight = train_sample_weights,
    eval_set = [(X_validate, y_validate)],
    verbose = True)

#bst = xgb.Booster({'nthread': 8})  # init model
#bst.load_model('model.bin')  # load data

# save model
#print('Saving model...')
#bst.save_model('poker_xgboost.model')

# make the prediction using the resulting model
print('Predicting...')
y_pred = bst.predict(X_test)

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

