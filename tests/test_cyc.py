from ModelTraining.feature_engineering.featureengineering.featurecreators import CyclicFeatures, CyclicFeaturesSampleTime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize('feature_creator',[
    CyclicFeatures(selected_feats=['weekday']),
    CyclicFeaturesSampleTime(sample_time=900, selected_feats=['weekday'])])


def test_cyc_feats_feature_addition(feature_creator):
    index = pd.date_range(pd.Timestamp(2021, 1, 1), pd.Timestamp(2021, 12, 31), freq='15T')
    col_name = 'Test_1'
    feat_names_full = [feature_creator._all_time_vals[name] for name in feature_creator.selected_feats]
    data = pd.DataFrame(index=index, data=np.zeros(len(index)), columns=[col_name])
    data_tr = feature_creator.transform(data)
    for feat_names in feat_names_full:
        for name in feat_names:
            assert (name in data_tr.columns)
            assert (name in feature_creator.get_additional_feat_names())
    assert (col_name in data_tr.columns)


if __name__ == "__main__":
    test_cyc_feats_feature_addition(CyclicFeatures(selected_feats=['weekday']))

    index = pd.date_range(pd.Timestamp(2021,1,1),pd.Timestamp(2021,12,31), freq='15T')
    data = pd.DataFrame(index=index, data=np.zeros(len(index)),columns=['Test_1'] )

    cyc_feats = CyclicFeatures(selected_feats=['weekday'])
    print(data.columns)
    data_tr = cyc_feats.transform(data)
    print(data_tr.columns)
    print(cyc_feats.get_additional_feat_names())
    cyc_feats_s = CyclicFeaturesSampleTime(sample_time=900)
    data_tr_s = cyc_feats_s.transform(data)

    plt.plot(data_tr['weekday_sin'][0:1000])
    plt.plot(data_tr['weekday_cos'][0:1000])
    plt.plot(data_tr_s['weekday_sin'][0:1000])
    plt.plot(data_tr_s['weekday_cos'][0:1000])
    plt.show()

    cyc_feats_week = CyclicFeatures(selected_feats=['week'])
    data_tr_w = cyc_feats_week.fit_transform(data)
    plt.plot(data_tr_w['week_sin'][0:10000])
    plt.show()

    plt.plot(data_tr['weekday_sin'][0:1000])
    plt.plot(data_tr['weekday_cos'][0:1000])
    plt.plot(data_tr_s['weekday_sin'][0:1000])
    plt.plot(data_tr_s['weekday_cos'][0:1000])
    plt.show()

