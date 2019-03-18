# https://github.com/tensorflow/adanet/blob/master/README.md

import adanet
import tensorflow as tf

# Define the model head for computing loss and evaluation metrics.
head = tf.contrib.estimator.multi_class_head(n_classes=10)

# Feature columns define how to process examples.
feature_columns = ...

# Learn to ensemble linear and neural network models.
estimator = adanet.AutoEnsembleEstimator(
    head=head,
    candidate_pool={
        "linear":
            tf.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=tf.train.FtrlOptimizer(...)),
        "dnn":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=tf.train.ProximalAdagradOptimizer(...),
                hidden_units=[1000, 500, 100])},
    max_iteration_steps=50)

estimator.train(input_fn=train_input_fn, steps=100)
metrics = estimator.evaluate(input_fn=eval_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)