import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# importing dataset
print('Importing datasets...')

class_labels = ['0','1','2','3','4','5','6','7','8','9']
predictor_labels = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS']

df_train = pd.read_csv('../Library/dataset/training.txt', header=None, sep=',')
df_test = pd.read_csv('../Library/dataset/testing.txt', header=None, sep=',')

dataset = pd.concat([df_train, df_test])

# Load the Diabetes Housing dataset
dataset.columns = predictor_labels

print(dataset.head())

print('Splitting the data into independent and dependent variables...')
X = dataset.iloc[:,0:10].values
y = dataset.iloc[:,10].values

# create training and testing vars
print('Creating training set and validation set...')
split_fraction = 1/math.sqrt(len(X[0]))
print(f'Splitting data with fraction {split_fraction}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_fraction, random_state = 21)

# Creating tree
print('Creating tree...')
classifier = RandomForestClassifier(
    verbose = 2,
    n_jobs = -1,
    criterion = 'entropy',
    n_estimators = 100,
    max_features = 'sqrt',
    max_depth = 30,
    min_samples_split = 10,
    min_samples_leaf = 1,
    bootstrap = False,
    random_state = 42)

# training
print('Training...')
classifier.fit(X_train, y_train)

# Saving model
print('Saving model...')
joblib.dump(classifier, 'randomforestmodel.pkl')

# Predicting
print('Predicting...')
y_pred = classifier.predict(X_test)

# Creating confusion matrix
print('Confusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Measuring accuracy
print('Accuracy:')
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

# Converting from probabillity to class
print('Plotting results...')

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()