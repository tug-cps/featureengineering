from ModelTraining.feature_engineering.featureengineering.featurecreators import CategoricalFeatures, CategoricalFeaturesDivider
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize('feature_creator',[
    CategoricalFeatures(selected_feats=['hour']),
    CategoricalFeaturesDivider(selected_feats=['hour'])])


def test_feature_addition(feature_creator):
    index = pd.date_range(pd.Timestamp(2021, 1, 1), pd.Timestamp(2021, 12, 31), freq='15T')
    col_name = 'Test_1'
    feat_names_full = [feature_creator._onehot_vals[name] for name in feature_creator.selected_feats]

    data = pd.DataFrame(index=index, data=np.zeros(len(index)), columns=[col_name])
    data_tr = feature_creator.transform(data)
    for feat_names in feat_names_full:
        for name in feat_names:
            assert(name in data_tr.columns)
            assert(name in feature_creator.get_additional_feat_names())
    assert(col_name in data_tr.columns)

def test_onehot_feature(feature_creator):
    index = pd.date_range(pd.Timestamp(2021, 1, 1), pd.Timestamp(2021, 12, 31), freq='15T')
    col_name = 'Test_1'
    data = pd.DataFrame(index=index, data=np.zeros(len(index)), columns=[col_name])
    data_tr = feature_creator.transform(data)
    assert(np.all(data_tr['hour_0'] == (data_tr.index.hour == 0)))


if __name__ == "__main__":
    test_feature_addition(CategoricalFeatures(selected_feats=['hour']))
    test_onehot_feature(CategoricalFeatures(selected_feats=['hour']))
    index = pd.date_range(pd.Timestamp(2021,1,1),pd.Timestamp(2021,12,31), freq='15T')
    data = pd.DataFrame(index=index, data=np.zeros(len(index)),columns=['Test_1'] )

    categorical_feats = CategoricalFeatures()
    data_c = categorical_feats.transform(data)
    plt.plot(data_c['mon'][0:1000])
    plt.plot(data_c['tue'][0:1000])
    plt.plot(data_c['wed'][0:1000])
    plt.show()
    print(categorical_feats.get_additional_feat_names())

    categorical_feats = CategoricalFeaturesDivider()
    data_cd = categorical_feats.transform(data)
    plt.plot(data_cd['hour_0'][0:100])
    plt.plot(data_cd['hour_2'][0:100])
    plt.plot(data_cd['hour_4'][0:100])
    plt.show()
    print(categorical_feats.get_additional_feat_names())
    print(data_cd.columns)
