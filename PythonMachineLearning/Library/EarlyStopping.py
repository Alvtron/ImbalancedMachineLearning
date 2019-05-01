from sklearn.base import ClassifierMixin, clone
from collections import defaultdict
import numpy as np
import time

class EarlyStopping(ClassifierMixin):
    def __init__(self, estimator, max_n_estimators, scorer, monitor_score = "gmean", patience=10, higher_is_better=False, sample_weight = None):
        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.scorer = scorer
        self.monitor_score = monitor_score
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.sample_weight = sample_weight
    
    def _make_estimator(self, append=True):
        """
        Make and configure a copy of the `estimator` attribute.
        
        Any estimator that has a `warm_start` option will work.
        """
        estimator = clone(self.estimator)
        estimator.n_estimators = 1
        estimator.warm_start = True
        return estimator
    
    def fit(self, X, y):
        """
        Fit `estimator` using X and y as training set.
        
        Fits up to `max_n_estimators` iterations and measures the performance
        on a separate dataset using `scorer`
        """
        est = self._make_estimator()
        self.eval_results = defaultdict(list)

        best_est = None
        self.best_score_ = 0
        self.best_iteration_ = 0
        iterations_since_last_score = 0

        for iteration in range(1, self.max_n_estimators + 1):
            start_time = time.time()
            est.n_estimators = iteration

            if (self.sample_weight is None):
                est.fit(X, y)
            else:
                est.fit(X, y, sample_weight = self.sample_weight)

            eval_metrics = self.scorer(est)

            # Create score string and save results
            score_string = ""
            for metric, score in eval_metrics.items():
                self.eval_results[metric].append(score)
                score_string += f' - {metric}: {score}'

            # Check if current estimator is better than the previous, save if true
            if (iteration is 1
                or ((self.higher_is_better and eval_metrics[self.monitor_score] > self.best_score_)
                or (not self.higher_is_better and eval_metrics[self.monitor_score] < self.best_score_))):
                self.best_score_ = eval_metrics[self.monitor_score]
                best_est = est
                self.best_iteration_ = iteration
                iterations_since_last_score = 0
                print(f"{iteration} of {self.max_n_estimators}- {round(time.time() - start_time)}s{score_string} (best)")
            else:
                print(f"{iteration} of {self.max_n_estimators}- {round(time.time() - start_time)}s{score_string}")

            # Increment and check the number of iterations since last best estimator
            iterations_since_last_score += 1
            if (iterations_since_last_score > self.patience):
                break;
        
        print(f"At iteration {self.best_iteration_}, the best overall score was {self.best_score_}")
        self.estimator = best_est
        return self