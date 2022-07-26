import numpy as np
import pandas as pd
from .maskfeats import MaskFeats


class MaskFeats_Inplace(MaskFeats):
    """
    Replace existing features with transformed features.
    """
    def __init__(self, features_to_transform=None):
        super().__init__(features_to_transform=features_to_transform)

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @param feature_names: feature names
        @return: full array
        """
        if self.features_to_transform is not None:
            # If transformation did not create new features, replace original by transformed values
            x_transf_new = X_orig.copy()
            indices = self.get_feat_indices(X_orig)
            if isinstance(X_orig, pd.DataFrame):
                x_transf_new[x_transf_new.columns[indices]] = X_transf
                if feature_names is not None:
                    column_names = np.array(x_transf_new.columns)
                    column_names[indices] = feature_names
                    x_transf_new.columns = column_names
            elif isinstance(X_orig, np.ndarray):
                x_transf_new[..., indices] = X_transf
            else:
                x_transf_new = np.array(x_transf_new)
                x_transf_new[indices] = X_transf
            return x_transf_new

        else:
            return X_transf