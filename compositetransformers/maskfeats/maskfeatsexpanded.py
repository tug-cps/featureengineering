import numpy as np
import pandas as pd
from .maskfeats import MaskFeats


class MaskFeats_Expanded(MaskFeats):
    """
    Expand existing feature vector
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
            # If transformation created new features: concatenate basic and new features
            if isinstance(X_orig, pd.DataFrame):
                x_basic = self.mask_feats(X_orig, inverse=True)
                dummy_feat_names = [f'feat_{i}' for i in range(x_basic.shape[-1], x_basic.shape[-1] + X_transf.shape[-1])]
                feat_names = dummy_feat_names if feature_names is None else feature_names
                for i, name in enumerate(feat_names):
                    x_basic[name] = X_transf[X_transf.columns[i]] if isinstance(X_transf, pd.DataFrame) else X_transf[..., i]
                return x_basic
            else:
                x_basic = self.mask_feats(X_orig, inverse=True)
                return np.concatenate((x_basic, X_transf), axis=-1)
        else:
            return X_transf