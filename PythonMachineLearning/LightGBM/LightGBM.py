from lightgbm import LGBMClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

# Importing dataset
poker = dataset.Poker([0.2, 0.2, 0.6], 0.01)

# Loading model
#print('Loading model')
#bst = lgb.Booster(model_file='model.txt')

# Creating model
model = LGBMClassifier(
    n_jobs = -1,
    n_estimators = 1000,
    num_leaves = 60,
    max_depth = 4,
    learning_rate = 1,
    objective = 'multiclass',
    metric  = 'multiclass',
    num_class = 10)

# Training
print('Training...')
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