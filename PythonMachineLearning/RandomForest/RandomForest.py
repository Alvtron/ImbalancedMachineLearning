import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Importing dataset
poker = dataset.Poker([0.2, 0.1, 0.7], 0.02, None, False)

# Setting parameters
parameters = {
    'verbose': 2,
    'n_jobs': -1,
    'random_state': 42,
    'bootstrap': True,
    'max_depth': 80,
    'max_features': 'auto',
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 600}

# Creating model
print('Creating tree...')
classifier = RandomForestClassifier(**parameters)

# Training
print('Training...')
classifier.fit(poker.X_train, poker.y_train, sample_weight = poker.train_sample_weights)

# Predicting
print('Predicting...')
y_pred = classifier.predict(poker.X_test)

# Analytics
title = "RandomForest ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(poker.sample_size * 100, poker.training_size * 100, poker.validation_size * 100, poker.testing_size * 100)
performance.write_parameters_to_file(title, parameters)
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = f'{title} - Confusion matrix', normalize = True)
plt.show()