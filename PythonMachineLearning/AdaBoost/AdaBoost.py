import MultiClassificationTrainer as mct
from Dataset import Poker
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

'''

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
on the original dataset and then fits additional copies of the classifier
on the same dataset but where the weights of incorrectly classified instances
are adjusted such that subsequent classifiers focus more on difficult cases.

This class implements the algorithm known as AdaBoost-SAMME.

'''

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

# Setting parameters, can be multiple
model_parameters = {
    'base_estimator': [DecisionTreeClassifier(max_depth = 10)], # The base estimator from which the boosted ensemble is built. Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes. If None, then the base estimator is DecisionTreeClassifier(max_depth=1)
    'n_estimators': [500], # 50, The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
    'learning_rate': [0.5], # 1.0, Learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learning_rate and n_estimators.
    'algorithm': ['SAMME.R'], # SAMME.R, If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
    'random_state': [42], # None, If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    }

# Training model
model, model_parameters, y_pred, gmean = mct.multiple_parameter_training(dataset, AdaBoostClassifier, model_parameters, True)

# Analytics
print('Analyzing...')
mct.analyze_and_save(
    title = "AdaBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f}) {4:0.15f}".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100, gmean),
    dataset = dataset,
    y_pred = y_pred,
    model_parameters = model_parameters,
    dataset_parameters = dataset_parameters)