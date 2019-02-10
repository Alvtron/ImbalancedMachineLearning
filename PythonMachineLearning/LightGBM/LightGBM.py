from lightgbm import LGBMClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

poker = dataset.Poker(0.5, 0.2)

# loading model
#print('Loading model')
#bst = lgb.Booster(model_file='model.txt')

# training
print('Training...')
model = LGBMClassifier(
    thread_count = 8,
    n_estimators = 2000,
    num_leaves = 60,
    depth = 4,
    learning_rate = 0.4,
    objective = 'multiclass',
    loss_function = 'MultiClass',
    eval_metric = 'MultiClass',
    classes_count = 10)
model.fit(X = poker.X_train, y = poker.y_train, sample_weight = poker.train_sample_weights, eval_set = [(poker.X_validate, poker.y_validate)], verbose = True)

# Saving model
#print('Saving model...')
#bst.save_model('model.txt')

# Predicting
print('Predicting...')
y_pred = model.predict(poker.X_test)

# Analytics
metric_results = model.evals_result_['valid_0']['multi_logloss']
print('Plotting evaluation metric results...')
performance.plot_evaluation_metric_results(metric_results, "LightGBM - Evaluation metric results")
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = 'LightGBM - Confusion matrix', normalize = True)
plt.show()