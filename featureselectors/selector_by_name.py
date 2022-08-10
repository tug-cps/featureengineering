import numpy as np
import pandas as pd
from . import FeatureSelector


class SelectorByName(FeatureSelector):
    """
    Selector by name:
    Select features by name
    """
    selected_feat_names = None
    feature_names_in = None

    def __init__(self, feature_names_in=[], selected_feat_names=[], **kwargs):
        super().__init__(**kwargs)
        self.feature_names_in = feature_names_in
        self.selected_feat_names = selected_feat_names

    def _fit(self, X, y, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in = X.columns
        return np.array([name in self.selected_feat_names for name in self.feature_names_in])

    def _get_support_mask(self):
        return np.array([name in self.selected_feat_names for name in self.feature_names_in])