import math
import numpy as np
import pandas as pd
from Performance import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool

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

pool_train = Pool(X_train, y_train, cat_features=class_labels)
pool_test = Pool(X_test, cat_features=class_labels)

#load_model("catboost_model", format='catboost')

model = CatBoostClassifier(
    task_type = 'GPU',
    thread_count = 8,
    num_trees = 10000,
    depth = 4,
    learning_rate = 0.29,
    loss_function = 'MultiClass',
    boosting_type = 'Plain',
    classes_count = 10,
    logging_level = 'Verbose')

model.fit(pool_train)

# save model
#print('Saving model...')
#model.save_model("catboost_model", format="cbm")

# make the prediction using the resulting model
print('Predicting...')
y_pred = model.predict(pool_test, prediction_type='Class',)

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