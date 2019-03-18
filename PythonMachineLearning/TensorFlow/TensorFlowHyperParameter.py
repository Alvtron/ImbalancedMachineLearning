# https://github.com/maxpumperla/hyperas
# https://towardsdatascience.com/keras-hyperparameter-tuning-in-google-colab-using-hyperas-624fa4bbf673
# https://www.kaggle.com/kt66nf/hyperparameter-optimization-using-keras-hyperas
# GPU https://research.wmz.ninja/articles/2017/01/configuring-gpu-accelerated-keras-in-windows-10.html
# https://www.pugetsystems.com/labs/hpc/The-Best-Way-to-Install-TensorFlow-with-GPU-Support-on-Windows-10-Without-Installing-CUDA-1187/

import MultiClassificationTrainer as mct
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, callbacks, backend, optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from Dataset import Poker

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def gmean(y_true, y_pred):
    """Compute the geometric mean.
    The geometric mean (G-mean) is the root of the product of class-wise
    sensitivity. This measure tries to maximize the accuracy on each of the
    classes while keeping these accuracies balanced.
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    recall = recall(y_true, y_pred)

    return backend.pow(tf.to_float(recall), 1.0/tf.size(recall))

def create_model(X_train, y_train, X_val, y_val):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7)])}}, input_shape=(len(X_train.columns),)))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense({{choice([np.power(2, 4), np.power(2, 5), np.power(2, 6)])}}))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5)])}}))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(10, activation='softmax'))
        
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    model.compile(
        optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
        loss={{choice(['categorical_crossentropy', 'sparse_categorical_crossentropy'])}},
        metrics=[f1])

    model.fit(
        X_train,
        y_train,
        epochs={{choice([25, 50, 75, 100])}},
        batch_size=2048,
        class_weight = {0:0.1995301487862177, 1:0.23664772727272726, 2:2.1035353535353534, 3:4.732954545454546, 4:25.48, 5:50.88018794048551, 6:69.41666666666667, 7:416.5, 8:7219.333333333333, 9:64974.0},
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr])

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def data():

    dataset_parameters = {
        'data_distribution': [0.2, 0.1, 0.7],
        'sample_size': 0.02,
        'sampling_strategy': None,
        'verbose': None}

    dataset = Poker(**dataset_parameters)

    X_train = dataset.X_train
    X_val = dataset.X_validate
    X_test = dataset.X_test
    y_train = dataset.y_train
    y_val = dataset.y_validate
    y_test = dataset.y_test

    return X_train, X_val, X_test, y_train, y_val, y_test

# ========== PROGRAM STARTS HERE ==========

# Improting dataset
dataset_parameters = {
    'data_distribution': [0.1, 0.1, 0.8],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': None}

# Hyper parameter training
print("Hyper parameter training...")
best_run, best_model = optim.minimize(model=create_model, data=data, functions=[f1], algo=tpe.suggest, max_evals=15, trials=Trials())

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

# Predict
print("Predicting...")
predictions = best_model.predict(dataset.X_test.values)
y_pred = []

for prediction in predictions:
    y_pred.append(np.argmax(prediction))

# Analytics
print('Analyzing...')
mct.analyze_and_save(
    title = "TensorFlow Keras ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(dataset.sample_size * 100, dataset.training_size * 100, dataset.validation_size * 100, dataset.testing_size * 100),
    dataset = dataset,
    y_pred = y_pred,
    metric_results = model.history.history,
    model_parameters = model_parameters,
    dataset_parameters = dataset_parameters)