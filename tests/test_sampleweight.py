from featureengineering.sampleweight import SampleWeight_DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def test_sampleweight():
    sampleweight = SampleWeight_DBSCAN(eps=1e-6)
    n_samples_cores = 1000
    n_samples_outliers = 200
    n_feats = 1
    cluster1 = np.ones((n_samples_cores,n_feats))
    cluster2 = np.zeros((n_samples_cores,n_feats))
    outliers =np.random.random_sample((n_samples_outliers, n_feats))
    x = np.concatenate((cluster1, cluster2,outliers))
    plt.figure()
    plt.hist(x)
    plt.show()

    sampleweight.fit(x)
    weights = sampleweight.weights_
    print(weights)
    print(sampleweight.get_num_core_samples())
    assert(sampleweight.get_num_core_samples() == 2 * n_samples_cores)


if __name__ == "__main__":
    test_sampleweight()