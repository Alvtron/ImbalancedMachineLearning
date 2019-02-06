import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# importing dataset
print('Importing datasets...')
df_train = pd.read_csv('training.txt', header=None, sep=',')
df_test = pd.read_csv('testing.txt', header=None, sep=',')

y_train = df_train[10]
y_test = df_test[10]
X_train = df_train.drop(10, axis=1)
X_test = df_test.drop(10, axis=1)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, y_test, reference=train_data)

# tuning parameters
print('Setting tuning parameters...')
parameters = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting': 'gbdt',
    'num_class':10,
    'num_trees':200,
    'learning_rate':0.05,
    'num_leaves':512
}

# training
print('Training...')
num_round = 20000
bst = lgb.train(parameters, train_data, num_round, valid_sets=[test_data])

# loading model
#print('Loading model')
#bst = lgb.Booster(model_file='model.txt')

# Saving model
print('Saving model...')
bst.save_model('model.txt')

# Predicting
print('Predicting...')
y_pred = bst.predict(X_test)

# Converting from probabillity to class
print('Converting probabillity chance to classes...')
predictions = []

for x in y_pred:
    predictions.append(np.argmax(x))

# Creating confusion matrix
print('Confusion matrix:')
cm = confusion_matrix(y_test, predictions)
print(cm)

# Measuring accuracy
print('Accuracy:')
accuracy=accuracy_score(predictions,y_test)
print(accuracy)

# Result: 72.7 % accuracy