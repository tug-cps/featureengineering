import numpy as np
import pandas as pd
from .maskfeats import MaskFeats


class MaskFeats_Expanded(MaskFeats):
    """
    Expand existing feature vector
    """
    def __init__(self, features_to_transform=None, **kwargs):
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
            x_basic = self.mask_feats(X_orig, inverse=True)
            if isinstance(X_orig, pd.DataFrame):
                x_basic = pd.DataFrame(index=X_orig.index) if x_basic is None else x_basic
                if feature_names is None:
                    feature_names = [f'feat_{i}' for i in
                                        range(x_basic.shape[-1], x_basic.shape[-1] + X_transf.shape[-1])]
                x_transf_vals = X_transf.values if isinstance(X_transf, pd.DataFrame) else X_transf
                x_transf_df = pd.DataFrame(index=X_orig.index, data=x_transf_vals, columns=feature_names)
                return x_transf_df if x_basic.shape[1] == 0 else x_basic.join(x_transf_df)
            else:
                return np.concatenate((x_basic, X_transf), axis=-1)
        else:
            return X_transf