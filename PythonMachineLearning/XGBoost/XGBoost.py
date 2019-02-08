import math
import numpy as np
import pandas as pd
import xgboost as xgb
from Performance import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# importing dataset
print('Importing dataset...')

predictor_labels = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS']

class_labels = [0,1,2,3,4,5,6,7,8,9]

class_descriptions = [
    '0: Nothing in hand; not a recognized poker hand',
    '1: One pair; one pair of equal ranks within five cards',
    '2: Two pairs; two pairs of equal ranks within five cards',
    '3: Three of a kind; three equal ranks within five cards',
    '4: Straight; five cards, sequentially ranked with no gaps',
    '5: Flush; five cards with the same suit',
    '6: Full house; pair + different rank three of a kind',
    '7: Four of a kind; four equal ranks within five cards',
    '8: Straight flush; straight + flush',
    '9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush'
    ]

df_train = pd.read_csv('../Library/dataset/training.txt', header=None, sep=',')
df_test = pd.read_csv('../Library/dataset/testing.txt', header=None, sep=',')
df_test.columns = predictor_labels
df_train.columns = predictor_labels

y_train = df_train['CLASS']
y_test = df_test['CLASS']
X_train = df_train.drop('CLASS', axis=1)
X_test = df_test.drop('CLASS', axis=1)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Setting parameters
print('Setting tuning parameters...')
param = {
    'max_depth': 4,
    'eta': 0.3,
    'silent': True,
    'objective': 'multi:softprob',
    'num_class': 10}

# 88.2547 %
#param = {
#    'max_depth': 4,
#    'eta': 0.3,
#    'silent': True,
#    'objective': 'multi:softprob',
#    'num_class': 10}

param['nthread'] = 8
param['eval_metric'] = 'auc'

# Training model
print('Training...')
num_round = 1000
bst = xgb.train(param, dtrain, num_round)

# Load model
#bst = xgb.Booster({'nthread': 8})  # init model
#bst.load_model('model.bin')  # load data

# save model
#print('Saving model...')
#bst.save_model('poker_xgboost.model')

# make the prediction using the resulting model
print('Predicting...')
y_pred = bst.predict(dtest)

# Converting from probabillity to class
print('Converting probabillity chance to classes...')
y_pred = np.asarray([np.argmax(line) for line in y_pred])

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