import numpy as np
from rdc import rdc
import pandas as pd
from . import FeatureSelectThreshold


class RDCThreshold(FeatureSelectThreshold):
    """
    R-Threshold:
    Threshold based on Random Dependence Coefficient value.
    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection threshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        if X.shape[-1] > 0:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], 1)
            return np.array([rdc(X[...,i], y) for i in range(X.shape[-1])])
        return np.zeros(1)