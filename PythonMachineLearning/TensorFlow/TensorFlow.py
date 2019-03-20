import MultiClassificationTrainer as mct
from Dataset import Poker
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM
from tensorflow.keras import backend, Sequential
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, Nadam, SGD, RMSprop
from tensorflow.keras.utils import plot_model
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self.validation_data[0])
        
        val_predict = []

        for prediction in predictions:
            val_predict.append(np.argmax(prediction))

        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return

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
        # Define epsilon so that the backpropagation will not result in NaN for 0 divisor case
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
    #'sampling_strategy': "SMOTE", # no significant improvement
    #'sampling_strategy': "over_and_under_sampling", # 10k and 20k shows promising for the first 8 classes, and 30-60% for class 9, but no hits on last class.
    #'sampling_strategy': "over_and_under_sampling_custom", # best result. 70% and 0% on two last classes, respectively.
    'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model = Sequential()
model.add(Dense(2 ** 10, input_shape=(dataset.X_train.shape[1],), name='input'))
model.add(Dense(2 ** 10, activation=tf.nn.relu, name='hidden1'))
model.add(Dense(2 ** 10, activation=tf.nn.relu, name='hidden2'))
model.add(Dense(2 ** 10, activation=tf.nn.relu, name='hidden3'))
model.add(Dense(10, activation=tf.nn.softmax, name='output'))

model_parameters = {
    #'optimizer':SGD(lr=0.0001, momentum=0.9), # slow
    #'optimizer':Adadelta(), # jiggles at 99%
    #'optimizer':Adamax(), # jiggles less at 99%
    'optimizer':Adam(lr=1e-5), # jiggles less at 99%, but spikes sometimes. More accurate.
    #'optimizer':Nadam(), # spikes a lot. Unuseable unless learning rate is adjusted.
    #'loss':categorical_focal_loss(gamma=2.0, alpha=0.25),
    #'loss':f1_loss,
    'loss':'sparse_categorical_crossentropy',
    'metrics':["accuracy"]
    }

model.compile(**model_parameters)

print(model.summary())

# Training
print("Training...")
callback = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto'),
    ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1)]
model.fit(
    dataset.X_train.values,
    dataset.y_train.values,
    batch_size=4096,
    epochs=1000,
    class_weight=dataset.weight_per_class,
    validation_data=(dataset.X_validate, dataset.y_validate),
    callbacks = callback)

# load a saved model
saved_model = load_model('best_model.h5')

# Predict
print("Predicting...")
predictions = saved_model.predict(dataset.X_test.values)
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