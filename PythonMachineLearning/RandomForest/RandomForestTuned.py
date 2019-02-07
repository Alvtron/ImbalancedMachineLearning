import math
import numpy as np
import pandas as pd
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
    n_iter = 100,
    cv = 3,
    verbose=2,
    random_state=42,
    n_jobs = -1)

# training
print('Training...')
classifier.fit(X_train, y_train)

classifier.best_params_

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