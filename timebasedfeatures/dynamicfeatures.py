import numpy as np
import scipy.signal as sig
from ..interfaces import BasicTransformer


class DynamicFeatures(BasicTransformer):
    """
    This class creates dynamic features with a certain lookback.
    Options:
        - flatten_dynamic_feats - all created features are added to the second dimension of the input data
        - return_3d_array: if flattening dynamic features, still a 3-d array can be returned through this option
    """
    lookback_horizon: int = 0
    flatten_dynamic_feats = False
    return_3d_array = False

    def __init__(self, lookback_horizon=5, flatten_dynamic_feats=False, return_3d_array=False, **kwargs):
        self.lookback_horizon = lookback_horizon
        self.flatten_dynamic_feats = flatten_dynamic_feats
        self.return_3d_array = return_3d_array

    def _transform(self, X):
        """
        Split into target segments
        @param X: input features (n_samples, n_features)
        @return: transformed features (n_samples, lookback + 1, n_features) or (n_samples, n_features)
        """
        X_transf = np.zeros((X.shape[0], self.lookback_horizon+1, X.shape[1]))
        # Tapped delay [0,0,0,0,1],[0,0,0,1,0],...
        bs = np.fliplr(np.identity(self.lookback_horizon + 1))
        # Filter signals
        for i in range(self.lookback_horizon+1):
            X_transf[:,i,:] = sig.lfilter(bs[i], 1.0, X, axis=0)
        if self.flatten_dynamic_feats:
            X_transf = X_transf.reshape(X_transf.shape[0], -1)
        if self.return_3d_array:
            if X_transf.ndim == 2:
                X_transf = X_transf.reshape(X_transf.shape[0], 1, X_transf.shape[1])
        return X_transf

    def _get_feature_names_out(self, feature_names=None):
        return [f'{name}_{lag}' for lag in np.flip(np.arange(1, self.lookback_horizon + 1)) for name in feature_names] + list(feature_names)
