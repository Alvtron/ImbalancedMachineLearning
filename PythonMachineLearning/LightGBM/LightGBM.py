import numpy as np
import lightgbm as lgb
import pandas as pd
from Performance import plot_confusion_matrix
from Dataset import create_poker_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights = create_poker_dataset()

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Setting parameters
print('Setting tuning parameters...')
parameters = {
    'objective': 'multiclass',
    "num_class" : 10,
    "num_leaves" : 60,
    "learning_rate" : 0.05
}

# training
print('Training...')
num_round = 200
bst = lgb.train(parameters, train_data, num_round, verbose_eval=True)

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
y_pred = np.asarray([np.argmax(line) for line in y_pred])

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