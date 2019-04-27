from sklearn.base import ClassifierMixin, clone
import numpy as np

class EarlyStopping(ClassifierMixin):
    def __init__(self, estimator, max_n_estimators, scorer, patience=10, higher_is_better=False):
        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.scorer = scorer
        self.patience = patience
        self.higher_is_better = higher_is_better
    
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
        self.scores_ = []

        best_est = None
        best_score = 0
        best_iteration = 0
        iterations_since_last_score = 0

        for iteration in range(1, self.max_n_estimators + 1):
            est.n_estimators = iteration
            est.fit(X, y)

            score = self.scorer(est)
            print(f"{iteration} of {self.max_n_estimators} - Score: {score}")
            self.scores_.append(score)

            if (iteration is 1
                or ((self.higher_is_better and score > best_score)
                or (not self.higher_is_better and score < best_score))):
                best_score = score
                best_est = est
                best_iteration = iteration
                iterations_since_last_score = 0

            iterations_since_last_score += 1
            if (iterations_since_last_score > self.patience):
                break;
        
        print(f"At iteration {best_iteration}, the best overall score was {best_score}")
        self.estimator = best_est
        return self