from sklearn.ensemble import AdaBoostClassifier
import Performance as performance
import Dataset as dataset
from pprint import pprint
from matplotlib import pyplot as plt

# AdaC2-based BQC2 classifier

# Importing dataset
poker = dataset.Poker([0.3, 0.7], 0.01, False)

params = {
    'n_estimators': 10,
    'learning_rate': 0.5,
    'algorithm': 'SAMME.R'}

# Loading model
#print('Loading existing model...')
#load_model("catboost_model", format='catboost')

# Creating model
print('Initializing model...')
model = AdaBoostClassifier(**params)

# Training
print('Training...')
model.fit(
    X = poker.X_train,
    y = poker.y_train,
    sample_weight = poker.train_sample_weights)

# Saving model
#print('Saving model...')
#model.save_model("catboost_model", format="cbm")

# Predicting
print('Predicting...')
y_pred = model.predict(poker.X_test)

# Analytics
print('Plotting evaluation metric results...')
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = 'CatBoost - Confusion matrix', normalize = True)
plt.show()