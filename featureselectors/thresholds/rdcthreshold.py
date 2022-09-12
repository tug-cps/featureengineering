import numpy as np
from rdc import rdc

from . import FeatureSelectThreshold


class RDCThreshold(FeatureSelectThreshold):
    """
    R-Threshold:
    Threshold based on Random Dependence Coefficient value.
    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        if X.shape[-1] > 0:
            return rdc(X, y)
        return np.zeros(1)