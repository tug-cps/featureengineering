import numpy as np
import pandas as pd
from .maskfeats import MaskFeats


class MaskFeats_Inplace(MaskFeats):
    """
    Replace existing features with transformed features.
    """
    rename_df_cols = True

    def __init__(self, features_to_transform=None, rename_df_cols=True, **kwargs):
        super().__init__(features_to_transform=features_to_transform)
        self.rename_df_cols = rename_df_cols

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @param feature_names: feature names
        @return: full array
        """
        if self.features_to_transform is None:
            return X_transf
        if len(self.features_to_transform) == 0:
            return X_orig
        indices = self.get_feat_indices(X_orig)
        x_new = self.replace_values(X_transf=X_transf, X_orig=X_orig, indices=indices)
        x_new = self.rename_cols(x_new, indices, feature_names)
        return x_new

    def replace_values(self, X_transf, X_orig, indices):
        x_new = X_orig.copy()
        if isinstance(X_orig, pd.DataFrame):
            x_new[x_new.columns[indices]] = X_transf
        else:
            x_new = np.array(x_new) if not isinstance(x_new, np.ndarray) else x_new
            x_new[..., indices] = X_transf
        return x_new

    def rename_cols(self, X, indices, feature_names=None):
        if isinstance(X, pd.DataFrame) and self.rename_df_cols and feature_names is not None:
            cols_to_rename = np.array(X.columns)[indices]
            return X.rename({col: name for col, name in zip(cols_to_rename, feature_names)}, axis=1)
        return X