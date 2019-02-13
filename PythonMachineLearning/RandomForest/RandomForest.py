import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Importing dataset
poker = dataset.Poker([0.2, 0.2, 0.6], 0.05)

# Creating model
print('Creating tree...')
classifier = RandomForestClassifier(
    verbose = 2,
    n_jobs = -1,
    random_state = 42,
    bootstrap = True,
    max_depth = 80,
    max_features = 'auto',
    min_samples_leaf = 1,
    min_samples_split = 5,
    n_estimators = 600)

# Training
print('Training...')
classifier.fit(poker.X_train, poker.y_train, sample_weight = poker.train_sample_weights)

# Saving model
#print('Saving model...')
#oblib.dump(classifier, 'randomforestmodel.pkl')

# Predicting
print('Predicting...')
y_pred = classifier.predict(poker.X_test)

# Analytics
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = 'RandomForest - Confusion matrix', normalize = True)
plt.show()