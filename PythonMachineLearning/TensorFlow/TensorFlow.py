import MultiClassificationTrainer as mct
from Dataset import Poker
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from imblearn.metrics import geometric_mean_score
import scipy as sp

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    recall = recall(y_true, y_pred)

    return K.pow(tf.to_float(recall), 1.0/tf.size(recall))

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_parameters = {
    'optimizer': 'adam', 
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
    }

model.compile(**model_parameters)

# Training
print("Training...")
model.fit(
    dataset.X_train.values,
    dataset.y_train.values,
    epochs=10,
    #class_weight = {0:0.1995301487862177, 1:0.23664772727272726, 2:2.1035353535353534, 3:4.732954545454546, 4:25.48, 5:50.88018794048551, 6:69.41666666666667, 7:416.5, 8:7219.333333333333, 9:64974.0},
    validation_data=(dataset.X_validate, dataset.y_validate))

# Predict
print("Predicting...")
predictions = model.predict(dataset.X_test.values)
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