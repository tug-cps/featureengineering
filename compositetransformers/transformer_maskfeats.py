import numpy as np
import pandas as pd
from . import TransformerWrapper, MaskFeats


class Transformer_MaskFeats(TransformerWrapper):
    """
    Transform only selected features - keep other features
    Transformed features replace basic features
    """
    feature_mask_ = None
    mask_params = {'features_to_transform': None}
    mask_type = "MaskFeats_Inplace"

    def __init__(self, transformer_type="", transformer_params={}, mask_type="MaskFeats_Inplace",mask_params={}, **kwargs):
        super().__init__(transformer_type=transformer_type, transformer_params=transformer_params)
        self.mask_type = mask_type
        self.mask_params = mask_params

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer.
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @param y: Target feature vector - optional
        @return: self
        """
        self._fit(self.mask_feats(X), y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform data
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        x_transf = self._transform(self.mask_feats(X))
        feat_names = self.transformer_.get_feature_names_out(self.mask_feats(X.columns)) if isinstance(X, pd.DataFrame) else None
        return self.feature_mask_.combine_feats(x_transf, X, feat_names)

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
        return self.feature_mask_.combine_feats(np.array(feature_names_tr), feature_names)

    def get_transformed_feature_names(self, feature_names=None):
        """
        Get transformed feature names
        @param feature_names: input feature names
        @return: transformed feature names
        """
        if feature_names is None:
            return None
        feat_names_to_transform = self.feature_mask_.mask_feats(feature_names)
        return self._get_feature_names_out(feat_names_to_transform)

    def mask_feats(self, X):
        """
        Mask features of X
        @param X: inputs to mask
        @return: masked inputs
        """
        self.feature_mask_ = MaskFeats.from_name(self.mask_type, **self.mask_params)
        return self.feature_mask_.mask_feats(X) if self.feature_mask_ is not None else X
