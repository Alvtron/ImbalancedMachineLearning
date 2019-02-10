import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

poker = dataset.Poker(0.5)

# Creating tree
print('Creating tree...')
classifier = RandomForestClassifier(
    verbose = 2,
    n_jobs = -1,
    random_state = 42,
    criterion = 'entropy',
    n_estimators = 1800,
    max_depth = 30,
    bootstrap = False,
    max_features = 'sqrt',
    min_samples_leaf = 1,
    min_samples_split = 10)

# training
print('Training...')
classifier.fit(poker.X_train, poker.y_train, sample_weight = poker.train_sample_weights)

# Saving model
#print('Saving model...')
j#oblib.dump(classifier, 'randomforestmodel.pkl')

# Predicting
print('Predicting...')
y_pred = classifier.predict(poker.X_test)

# Analytics
metric_results = model.get_evals_result()['learn']['MultiClass']
print('Plotting evaluation metric results...')
performance.plot_evaluation_metric_results(metric_results, "CatBoost - Evaluation metric results")
performance.print_advanced_metrics(y_pred, poker.y_test, dataset.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, dataset.class_labels, title = 'CatBoost - Confusion matrix', normalize = True)
plt.show()