from featureengineering.timebasedfeatures import StatisticalFeaturesNumpy
import numpy as np
import pandas as pd


def test_statfeats():
    X = np.random.randn(100,3)
    statfeats = StatisticalFeaturesNumpy(window_size=4,statistical_features=['min'])
    X_tr = statfeats.fit_transform(X)
    assert(X_tr.shape==X.shape)
    for feat in range(3):
        for win in range(25):
            assert(np.all(X_tr[win*4:win*4+4,feat]==np.min(X[win*4:win*4+4,feat],axis=0)))

def test_statfeats_df():
    X = pd.DataFrame(data=np.random.randn(80,3), columns=['Feat_0','Feat_1','Feat_2'])
    statfeats = StatisticalFeaturesNumpy(window_size=4,statistical_features=['min','max'])
    X_tr = statfeats.fit_transform(X)
    for i in range(3):
        assert(np.all(X_tr[f'Feat_{i}_min_4'][:4]==np.min(X[f'Feat_{i}'][:4],axis=0)))
        assert(np.all(X_tr[f'Feat_{i}_min_4'][4:8] == np.min(X[f'Feat_{i}'][4:8], axis=0)))
        assert(np.all(X_tr[f'Feat_{i}_max_4'][:4]==np.max(X[f'Feat_{i}'][:4],axis=0)))
        assert(np.all(X_tr[f'Feat_{i}_max_4'][4:8] == np.max(X[f'Feat_{i}'][4:8], axis=0)))


if __name__ == "__main__":
    test_statfeats()
    test_statfeats_df()