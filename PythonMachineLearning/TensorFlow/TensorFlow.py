import MultiClassificationTrainer as mct
from Dataset import Poker
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM
from tensorflow.keras import backend, Sequential
from tensorflow.keras.optimizers import SGD
from imblearn.metrics import geometric_mean_score
import scipy as sp

def f1(y_true, y_pred):
    y_pred = backend.round(y_pred)
    tp = backend.sum(backend.cast(y_true*y_pred, 'float'), axis=0)
    tn = backend.sum(backend.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = backend.sum(backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2*p*r / (p+r+backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return backend.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = backend.sum(backend.cast(y_true*y_pred, 'float'), axis=0)
    tn = backend.sum(backend.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = backend.sum(backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2*p*r / (p+r+backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - backend.mean(f1)

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

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Paper:
        https://arxiv.org/abs/1708.02002
    Parameters:
        alpha -- the same as weighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = backend.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = backend.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*backend.log(y_pred)
        # Calculate weight that consists of modulating factor and weighting factor
        weight = alpha * y_true * backend.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = backend.sum(loss, axis=1)
        return loss
    
    return focal_loss

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    'sampling_strategy': "over_and_under_sampling",
    'verbose': True}

dataset = Poker(**dataset_parameters)

model = Sequential()
model.add(Dense(512, input_shape=(dataset.X_train.shape[1],), name='input'))
model.add(Dense(512, activation=tf.nn.relu, name='hidden1'))
model.add(Dense(512, activation=tf.nn.relu, name='hidden2'))
model.add(Dense(512, activation=tf.nn.relu, name='hidden3'))
model.add(Dense(512, activation=tf.nn.relu, name='hidden4'))
model.add(Dense(10, activation=tf.nn.softmax, name='output'))

model_parameters = {
    #'optimizer':SGD(lr=0.1, momentum=0.9),
    'optimizer':'adam',
    #'loss':categorical_focal_loss(gamma=2.0, alpha=0.25),
    'loss':'sparse_categorical_crossentropy',
    'metrics': ["accuracy", f1]
    }

model.compile(**model_parameters)

# Training
print("Training...")
model.fit(
    dataset.X_train.values,
    dataset.y_train.values,
    batch_size=8192,
    epochs=20,
    validation_data=(dataset.X_validate, dataset.y_validate))

print(model.summary())

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