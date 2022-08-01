import numpy as np
import pandas as pd
from .maskfeats import MaskFeats


class MaskFeats_Addition(MaskFeats):
    """
    Add new features at the end of the feature vector.
    """

    def __init__(self, features_to_transform=None, **kwargs):
        super().__init__(features_to_transform=features_to_transform)

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @return: full array
        """
        if self.features_to_transform is not None:
            # If transformation did not create new features, replace original by transformed values
            x_transf_new = X_orig.copy()
            if isinstance(X_orig, pd.DataFrame):
                dummy_feat_names = [f'feat_{i}' for i in range(X_orig.shape[-1], X_orig.shape[-1] + X_transf.shape[-1])]
                feat_names_new = dummy_feat_names if feature_names is None else feature_names
                x_transf_new[feat_names_new] = X_transf
            else:
                x_transf_new = np.concatenate((np.array(X_orig), np.array(X_transf)))
            return x_transf_new
        else:
            return X_transf