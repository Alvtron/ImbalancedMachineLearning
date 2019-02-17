import CatBoost
import XGBoost
import LightGBM
import RandomForest

# Importing dataset
poker = dataset.Poker([0.2, 0.1, 0.7], 0.02, None, True)

y_pred, metric_results = CatBoost.train(
    X_train = poker.X_train,
    y_train = poker.y_train,
    X_test = poker.X_test,
    y_test = poker.y_test,
    eval_set = [(poker.X_validate, poker.y_validate)],
    sample_weights = poker.train_sample_weights,
    verbose = True)

# Analytics
print('Plotting evaluation metric results...')
title = "CatBoost ({0:0.0f}% sample {1:0.0f}-{2:0.0f}-{3:0.0f})".format(poker.sample_size * 100, poker.training_size * 100, poker.validation_size * 100, poker.testing_size * 100)
performance.plot_evaluation_metric_results(metric_results, f'{title} - Evaluation metrics')
performance.print_advanced_metrics(y_pred, poker.y_test, poker.class_labels, poker.class_descriptions)
performance.plot_confusion_matrix(y_pred, poker.y_test, poker.class_labels, title = f'{title} - Confusion matrix', normalize = True)
plt.show()