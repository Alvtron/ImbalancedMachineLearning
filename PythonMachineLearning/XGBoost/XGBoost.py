import xgboost as xgb
from xgboost import plot_tree
from xgboost import XGBClassifier
import Performance as performance
import Dataset as dataset
from matplotlib import pyplot as plt

poker = dataset.Poker(0.6, 0.4)

print('Creating model with tuning parameters...')
model = XGBClassifier(
    nthread = 8,
    n_jobs = -1,
    num_class = 10,
    n_estimators= 1000,
    max_depth = 4,
    learning_rate = 0.5,
    eval_metric = 'mlogloss',
    objective = 'multi:softmax')

print('Training...')
model.fit(
    X = poker.X_train,
    y = poker.y_train,
    sample_weight = poker.train_sample_weights,
    eval_set = [(poker.X_validate, poker.y_validate)],
    verbose = True)

#model = xgb.Booster({'nthread': 8})  # init model
#model.load_model('model.bin')  # load data

# save model
#print('Saving model...')
#model.save_model('poker_xgboost.model')

# make the prediction using the resulting model
print('Predicting...')
y_pred = model.predict(poker.X_test)

# Analytics
metric_results = model.evals_result()['validation_0']['mlogloss']
print('Plotting evaluation metric results...')
performance.plot_evaluation_metric_results(metric_results, "XGBoost - Evaluation metric results")
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = 'XGBoost - Confusion matrix', normalize = True)
plt.show()

