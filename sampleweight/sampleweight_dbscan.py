from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator
import numpy as np


class SampleWeight_DBSCAN(BaseEstimator):
    """
    Create sample weights based on DBSCAN algorithm.
    """
    eps = 3
    weight_core_samples = 1
    weight_outlier_samples = 0
    weights_ = None
    dbscan_ = None

    def __init__(self, eps=3, weight_core_samples=1, weight_outlier_samples=0, **kwargs):
        self.eps = eps
        self.weight_core_samples = weight_core_samples
        self.weight_outlier_samples = weight_outlier_samples

    def fit(self, X, y=None, **fit_params):
        self.dbscan_ = DBSCAN(eps=self.eps)
        self.dbscan_.fit(X, y, **fit_params)
        self.create_weights(X.shape[0])

    def create_weights(self, n_samples_total=0):
        self.weights_ = np.ones(n_samples_total) * self.weight_outlier_samples
        self.weights_[self.dbscan_.core_sample_indices_] = self.weight_core_samples

    def get_num_core_samples(self):
        if self.dbscan_ is not None and hasattr(self.dbscan_, "core_sample_indices_"):
            return len(self.dbscan_.core_sample_indices_)
        return None