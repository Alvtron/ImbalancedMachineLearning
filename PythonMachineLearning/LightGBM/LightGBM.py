from lightgbm import LGBMClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

poker = dataset.Poker(0.6, 0.4)

# loading model
#print('Loading model')
#bst = lgb.Booster(model_file='model.txt')

# training
print('Training...')
model = LGBMClassifier(
    n_estimators = 1000,
    num_leaves = 60,
    max_depth = 4,
    learning_rate = 1,
    objective = 'multiclass',
    metric  = 'multiclass',
    num_class = 10)
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