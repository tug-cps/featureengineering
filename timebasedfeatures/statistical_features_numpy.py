import pandas as pd
import numpy as np
from ..interfaces import BasicTransformer


class StatisticalFeaturesNumpy(BasicTransformer):
    """
    Statistical features.
    Parameters: Window size, features to select
    """
    _all_stat_feats = ['mean', 'std', 'max', 'min']
    window_size = 2
    statistical_features = []

    def __init__(self, statistical_features=['min', 'max'], window_size=2, **kwargs):
        self.statistical_features = statistical_features
        self.window_size = window_size

    def transform(self, X):
        X_to_transform = X.values if isinstance(X,pd.DataFrame) else X
        X_tr = self._transform(X_to_transform)
        if isinstance(X, pd.DataFrame):
            X_tr = pd.DataFrame(index=X.index, data=X_tr, columns=self.get_feature_names_out(X.columns))
        return X_tr

    def _transform(self, X):
        """
        Add statistical features
        @param X: input data
        @return: transformed data
        """
        num_full_windows = int(X.shape[0] / self.window_size)
        X_to_reshape = X[:num_full_windows * self.window_size]
        X_reshaped = np.reshape(X_to_reshape, (num_full_windows, self.window_size, X.shape[1]))

        X_tr_reshaped = np.concatenate([np.repeat(getattr(np, feat)(X_reshaped, axis=1), self.window_size, axis=0)
                                        for feat in self.statistical_features], axis=-1)
        X_rest = X[num_full_windows * self.window_size:]
        if X_rest.shape[0] > 0:
            X_tr_rest = np.zeros((X_rest.shape[0], len(self.statistical_features) * X_rest.shape[1]))
            for i, feat in enumerate(self.statistical_features):
                X_tr_rest[:,i*X_rest.shape[1]:(i+1)*X_rest.shape[1]] = getattr(np, feat)(X_rest, axis=0)
            X_tr = np.concatenate((X_tr_reshaped, X_tr_rest), axis=0)
        else:
            X_tr = X_tr_reshaped
        return X_tr

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_{stat_name}_{self.window_size}" for stat_name in self.statistical_features for name in feature_names]