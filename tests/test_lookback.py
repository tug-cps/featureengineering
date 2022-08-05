from featureengineering.timebasedfeatures.dynamicfeatures import DynamicFeatures
import numpy as np
import pandas as pd


def test_lookback_basic():
    n_feats = 3
    X = np.random.randn(100,n_feats)
    lookback_horizon = 1
    tr = DynamicFeatures(lookback_horizon=lookback_horizon)
    X_tr = tr.fit_transform(X)
    assert(X_tr.ndim ==3)
    assert(X_tr.shape == (X.shape[0], lookback_horizon + 1, X.shape[1]))
    assert(np.all(X_tr[:,1,:] == X))
    X_delayed = np.concatenate((np.zeros((1,n_feats)), X[:-1]))
    assert(np.all(X_tr[:,0,:] == X_delayed))


def test_lookback_flatten():
    n_feats = 3
    X = np.random.randn(100, n_feats)
    lookback_horizon = 1
    tr = DynamicFeatures(lookback_horizon=lookback_horizon, flatten_dynamic_feats=True)
    X_tr = tr.fit_transform(X)
    assert (X_tr.ndim == 2)
    assert (X_tr.shape == (X.shape[0], (lookback_horizon + 1) * X.shape[1]))


def test_lookback_df():
    n_feats = 3
    columns = [f"feat_{i}" for i in range(n_feats)]
    X = pd.DataFrame(data=np.random.randn(100, n_feats), columns=columns)
    lookback_horizon = 1
    tr = DynamicFeatures(lookback_horizon=lookback_horizon, flatten_dynamic_feats=True)
    X_tr = tr.fit_transform(X)
    assert (X_tr.ndim == 2)
    assert (X_tr.shape == (X.shape[0], (lookback_horizon + 1) * X.shape[1]))
    assert(np.all(X['feat_0'] == X_tr['feat_0']))
    X_delayed = np.concatenate(([0], X['feat_0'][:-1]))
    assert(np.all(X_delayed == X_tr['feat_0_1']))


def test_lookback_3d():
    n_feats = 3
    X = np.random.randn(100, n_feats)
    lookback_horizon = 1
    tr = DynamicFeatures(lookback_horizon=lookback_horizon, flatten_dynamic_feats=True, return_3d_array=True)
    X_tr = tr.fit_transform(X)
    assert (X_tr.ndim == 3)
    assert (X_tr.shape == (X.shape[0],1, (lookback_horizon + 1) * X.shape[1]))


if __name__ == "__main__":
    test_lookback_basic()
    test_lookback_flatten()
    test_lookback_3d()
    test_lookback_df()