from Evaluation import Evaluator
from Dataset import Poker
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, cross_val_score, PredefinedSplit
from sklearn.linear_model import SGDClassifier
import time

def plot_evaluation_metric_results(metric_results):
    plt.figure()
    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title('Evaluation metrics')
    plt.axhline(y = 0, linewidth=0.5, color = 'k')
    for type, result in flatten(metric_results, sep='_').items():
        line, = plt.plot(result, label=f"{type}")
        plt.legend()

# Create custom metric
gmean_scorer = make_scorer(score_func=geometric_mean_score, greater_is_better=True)

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE", # no significant improvement
    #'sampling_strategy': "over_and_under_sampling", # 10k and 20k shows promising for the first 8 classes, and 30-60% for class 9, but no hits on last class.
    #'sampling_strategy': "over_and_under_sampling_custom", # best result. 70% and 0% on two last classes, respectively.
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model_parameters = {
    'loss':'hinge', # The loss function to be used. Defaults to ‘hinge’, which gives a linear SVM.
    'penalty':'l2', # The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.
    'alpha':0.0001, # Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.
    'l1_ratio':0.15, # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
    'fit_intercept':True, # Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.
    'max_iter':1000, # The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit. Defaults to 5. Defaults to 1000 from 0.21, or if tol is not None.
    'average':False, # When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute. If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.
    'class_weight':dataset.weight_per_class, # Preset for the class_weight fit parameter. Weights associated with classes. If not given, all classes are supposed to have weight one.
    'early_stopping':False, # Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
    'epsilon':0.1, # Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
    'eta0':0.0, # The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
    'learning_rate':'optimal', # The learning rate schedule: 'constant': eta = eta0 'optimal': [default] eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou. 'invscaling': eta = eta0 / pow(t, power_t) 'adaptive': eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.
    'n_iter':None, # The number of passes over the training data (aka epochs). Defaults to None. Deprecated, will be removed in 0.21.
    'n_iter_no_change':5, # Number of iterations with no improvement to wait before early stopping.
    'n_jobs':-1, # The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation.
    'power_t':0.5, # The exponent for inverse scaling learning rate [default 0.5].
    'random_state':42, # The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator
    'shuffle':True, # Whether or not the training data should be shuffled after each epoch. Defaults to True.
    'tol':0.001, # The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to None. Defaults to 1e-3 from 0.21.
    'validation_fraction':0.1, # The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.
    'verbose':1, # The verbosity level
    'warm_start':False # When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. See the Glossary. Repeatedly calling fit or partial_fit when warm_start is True can result in a different solution than when calling fit a single time because of the way the data is shuffled. If a dynamic learning rate is used, the learning rate is adapted depending on the number of samples already seen. Calling fit resets this counter, while partial_fit will result in increasing the existing counter.
    }

sgd = SGDClassifier(**model_parameters)

# Training
print('Training...')
start_time = time.time()
scores = cross_validate(
    estimator=sgd,
    X=np.concatenate((dataset.X_train.values, dataset.X_validate.values)),
    y=np.concatenate((dataset.y_train.values, dataset.y_validate.values)),
    cv=PredefinedSplit(test_fold=[-1] * len(dataset.X_train.values) + [0] * len(dataset.X_validate.values)),
    scoring=gmean_scorer,
    return_train_score=True,
    verbose=True,
    n_jobs=-1)
elapsed_time_training = time.time() - start_time

#sgd.fit(X=dataset.X_train.values, y=dataset.y_train.values)

# Predict
#print("Predicting...")
#y_pred = sgd.predict(dataset.X_test.values)

# Analytics
#print('Analyzing...')
#mct.analyze_and_save(
#    title = "ThunderSVM ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100),
#    dataset = dataset,
#    y_pred = y_pred,
#    metric_results = scores,
#    model_parameters = model_parameters,
#    dataset_parameters = dataset_parameters)

plot_evaluation_metric_results(scores)
plt.show()