from Evaluation import Evaluator
from Dataset import Poker
from TensorFlowMultiClassMetrics import precision, recall, f1
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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import time

# Doesnt work
def gmean(y_true, y_pred):
    """Compute the geometric mean.
    The geometric mean (G-mean) is the root of the product of class-wise
    sensitivity. This measure tries to maximize the accuracy on each of the
    classes while keeping these accuracies balanced.

    Papers

    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of
       imbalanced training sets: one-sided selection" ICML (1997)

    .. [2] Barandela, R., Sánchez, J. S., Garcıa, V., & Rangel, E. "Strategies
       for learning in class imbalance problems", Pattern Recognition,
       36(3), (2003), pp 849-851.

    """

    def recall(y_true, y_pred):
        y_pred = backend.round(y_pred)
        tp = backend.sum(backend.cast(y_true*y_pred, tf.float32), axis=0)
        fp = backend.sum(backend.cast((1-y_true)*y_pred, tf.float32), axis=0)
        fn = backend.sum(backend.cast(y_true*(1-y_pred), tf.float32), axis=0)
        return tp / (tp + fn + backend.epsilon())

    def element_wise_recall(y_true, y_pred):
        y_pred = backend.round(y_pred)
        tp = backend.cast(y_true*y_pred, tf.float32)
        fp = backend.cast((1 - y_true)*y_pred, tf.float32)
        fn = backend.cast(y_true*(1 - y_pred), tf.float32)
        return tp / (tp + fn + backend.epsilon())

    def number_of_classes(y_pred):
        value = backend.shape(y_pred)[1]
        return tf.cond(
            tf.equal(value, 0),
            lambda: tf.constant(0, tf.int32),
            lambda: value)

    # Create empty recall list
    recalls = tf.constant(1.0, shape=[0,10])

    def multiply_recalls(x):
        X = tf.cond(
            tf.equal(x[0], x[1]),
            lambda: tf.constant(1, tf.int32),
            lambda: tf.constant(0, tf.int32))
        y = 1
        r = element_wise_recall(y, X)
        indices = x[0]
        tf.scatter_add(recalls, indices, r)

    # flatten y_true
    y_true = tf.reshape(y_true, [-1])
    # get number of classes
    num_classes = number_of_classes(y_pred)
    # class predictions
    y_pred_classes = backend.map_fn(lambda x: backend.argmax(x), y_pred)
    # Concat
    y_true_y_pred = tf.stack([y_true, y_pred_classes])
    # create recall value per class
    backend.map_fn(lambda x: multiply_recalls(x), y_true_y_pred)
    # Multiply recall values
    recall_value = backend.prod(recalls)
    # create exponent
    b = tf.constant(1, tf.float32) / num_classes
    result = tf.pow(recall_value, b)

    with tf.Session() as sess:
        return sess.run(result)
 
def target_shape_rows(y_true, y_pred):
    value = backend.shape(y_true)[0]
    return tf.cond(
        tf.equal(value, 0),
        lambda: tf.constant(0, tf.int32),
        lambda: value)
    

def target_shape_columns(y_true, y_pred):
    value = backend.shape(y_true)[1]
    return tf.cond(
        tf.equal(value, 0),
        lambda: tf.constant(0, tf.int32),
        lambda: value)

def prediction_shape_rows(y_true, y_pred):
    value = backend.shape(y_pred)[0]
    return tf.cond(
        tf.equal(value, 0),
        lambda: tf.constant(0, tf.int32),
        lambda: value)

def prediction_shape_columns(y_true, y_pred):
    value = backend.shape(y_pred)[1]
    return tf.cond(
        tf.equal(value, 0),
        lambda: tf.constant(0, tf.int32),
        lambda: value)

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

class EarlyStopByGmean(Callback):

    def __init__(self, verbose = False, patience = 25, save_path = None):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.save_path = save_path
        self.patience = patience
        self.best_score = 0.0
        self.best_epoch = 0
        self.val_gmean = []
        self.val_accuracy = []
        
        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.targets = []
        self.predictions = []
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        self.targets.extend(np.ravel(backend.eval(self.var_y_true)))
        self.predictions.extend(backend.eval(self.var_y_pred))

    def on_epoch_end(self, epoch, logs=None):
        # fetch results
        targets = self.targets
        predictions = self.predictions
        # convert prediction class probabilities to class
        y_pred = np.asarray([np.argmax(line) for line in predictions])
        # calculate metrics with sklearn
        gmean = geometric_mean_score(targets, y_pred, average = 'macro')
        accuracy = accuracy_score(targets, y_pred)
        # save scores
        self.val_gmean.append(gmean)
        self.val_accuracy.append(accuracy)
        # reset results
        self.targets = []
        self.predictions = []
        # check results
        if (gmean > self.best_score):
            self.best_score = gmean
            self.best_epoch = epoch
            self.model.save(self.save_path)
            if (self.verbose is True):
                print(f"{epoch} - gmean: {gmean} - accuracy: {accuracy} (best)")
        else:
            if (self.verbose is True):
                print(f"{epoch} - gmean: {gmean} - accuracy: {accuracy}")
        # end if patience is overdue
        if (epoch - self.patience > self.best_epoch):
            if (self.verbose is True):
                print(f"Epoch {epoch}: early stopping Threshold")
            self.model.stop_training = True

# Importing dataset
dataset_parameters = {
    'data_distribution': [0.2, 0.1, 0.7],
    'sample_size': 0.02,
    #'sampling_strategy': "SMOTE", # no significant improvement
    #'sampling_strategy': "over_and_under_sampling", # 10k and 20k shows promising for the first 8 classes, and 30-60% for class 9, but no hits on last class.
    #'sampling_strategy': "over_and_under_sampling_custom", # best result. 70% and 0% on two last classes, respectively.
    'sampling_strategy': "WSMOTE", # 
    #'sampling_strategy': None,
    'verbose': False}

dataset = Poker(**dataset_parameters)

model = Sequential()
model.add(Dense(2 ** 10, input_shape=(dataset.X_train.shape[1],), name='input'))

num_hidden_layers = 3
neuron_base = 2
neuron_exponent = 10
neuron_delta = 0

for layer in range(0, num_hidden_layers):
    model.add(Dense(
        neuron_base ** (neuron_exponent + (neuron_delta * layer)),
        activation=tf.nn.relu,
        name=f'hidden{layer + 1}'))

model.add(Dense(10, activation=tf.nn.softmax, name='output'))

model_parameters = {
    #'optimizer':SGD(lr=0.0001, momentum=0.9), # slow
    #'optimizer':Adadelta(), # jiggles at 99%
    #'optimizer':Adamax(), # jiggles less at 99%
    'optimizer':Adam(lr=1e-5), # jiggles less at 99%, but spikes sometimes. More accurate.
    #'optimizer':Nadam(), # spikes a lot. Unuseable unless learning rate is adjusted.
    #'loss':categorical_focal_loss(gamma=4.0, alpha=0.25),
    #'loss':f1_loss,
    'loss':'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],
    }

model.compile(**model_parameters)

print(model.summary())

# Training
print('Training...')
start_time = time.time()
callback = [EarlyStopByGmean(verbose = True, patience=25, save_path='best_model.h5')]
#EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto'),
#ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

# Assign training data results to callable variable
fetches = [
    tf.assign(callback[0].var_y_true, model.targets[0], validate_shape=False),
    tf.assign(callback[0].var_y_pred, model.outputs[0], validate_shape=False)]
model._function_kwargs = {'fetches': fetches}

model.fit(
    dataset.X_train.values,
    dataset.y_train.values,
    batch_size=4096,
    epochs=100000,
    class_weight=dataset.weight_per_class,
    validation_data=(dataset.X_validate, dataset.y_validate),
    callbacks = callback)

elapsed_time_training = time.time() - start_time

# load a saved model
saved_model = load_model('best_model.h5')

# Predicting
print('Predicting...')
start_time = time.time()
predictions = saved_model.predict(dataset.X_test.values)
elapsed_time_testing = time.time() - start_time

y_pred = np.asarray([np.argmax(line) for line in predictions])

# Analytics
eval_results = {
    'accuracy': callback[0].val_accuracy,
    'gmean': callback[0].val_gmean}
title = "TensorFlow (4 layers - weights - WSMOTE)"
save_path = "C:/Users/thoma/source/repos/PythonMachineLearning/PythonMachineLearning/Library/Results"
print('Analyzing...')
evaluator = Evaluator(title, save_path)
evaluator.append_to_file(f'Best iteration: {callback[0].best_epoch}', "info.txt")
evaluator.append_to_file(f'Training time (seconds): {elapsed_time_training}', "info.txt")
evaluator.append_to_file(f'Testing time (seconds): {elapsed_time_testing}', "info.txt")
evaluator.append_to_file(model.get_config(), "model_config.txt")
evaluator.append_to_file(model.summary(), "model_summary.txt")
evaluator.append_to_file(dataset_parameters, "dataset_parameters.txt")
evaluator.append_to_file(model_parameters, "model_parameters.txt")
evaluator.save_advanced_metrics(dataset.y_test, y_pred, dataset.class_labels, dataset.class_descriptions)
evaluator.append_to_file(eval_results, "metric_results.txt")
evaluator.create_evaluation_metric_results(eval_results, xlabel='epochs', ylabel='metric score')
evaluator.create_confusion_matrix(dataset.y_test, y_pred, dataset.class_labels, normalize = True)
plt.show()