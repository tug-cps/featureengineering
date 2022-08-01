import numpy as np
import pandas as pd
from ...interfaces import BasicInterface, FeatureNames


class MaskFeats(FeatureNames, BasicInterface):
    """
    Mask Features
    """
    features_to_transform = None

    def __init__(self, features_to_transform=None, **kwargs):
        self.features_to_transform = features_to_transform

    def get_feat_indices(self, X, inverse=False):
        """
        Get feature indices of mask - numpy boolean array index
        @param X: vector containing features - np.array or pd dataframe
        @param inverse: get selected features or non-selected features
        @return: array of booleans
        """
        if self.features_to_transform is not None and len(self.features_to_transform) > 0:
            if isinstance(self.features_to_transform[0], str):
                feat_ind = np.array([feat in self.features_to_transform for feat in (X.columns if isinstance(X, pd.DataFrame) else X)])
                return feat_ind if not inverse else np.bitwise_not(feat_ind)
            else:
                return self.features_to_transform if not inverse else np.bitwise_not(self.features_to_transform)
        return None

    def mask_feats(self, X, inverse=False):
        """
        Select features to transformation
        @param X: all features
        @param inverse: invert features_to_transform
        @return: selected features
        """
        mask = self.get_feat_indices(X, inverse)
        if mask is not None:
            if isinstance(X, pd.DataFrame):
                return X[X.columns[mask]]
            else:
                X = np.array(X) if not isinstance(X, np.ndarray) else X
                return X[..., mask]
        else:
            return None if inverse else X

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine original and transformed features - override this method
        @param X_orig: original feature vector
        @param X_transf: transformed feature vector
        @param feature_names: Feature names
        @return: combined features
        """
        return X_transf

    def get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: transformed feature names
        """
        if feature_names is None:
            return None
        feat_names_to_transform = self.mask_feats(feature_names)
        feature_names_tr = self._get_feature_names_out(feat_names_to_transform)
        return self.combine_feats(np.array(feature_names_tr), feature_names)


