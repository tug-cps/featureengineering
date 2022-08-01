from featureengineering.filters import ButterworthFilter, Filter
from pathlib import Path
import os

def test_filter_params_baseclass():
    filter_1 = ButterworthFilter(T=20, order=3)
    params_but = filter_1.get_params(deep=True)
    filter_basic = Filter()
    params_basic = filter_basic.get_params(deep=True)
    # Are params different from each other?
    assert(params_but != params_basic)
    # Are basic params in subclass params?
    assert([param in params_but for param in params_basic])


def test_store_load_filter():
    path = os.path.join(Path(__file__).parent, "test_output")
    filename = "LPF.pickle"

    filter_1 = ButterworthFilter(T=20, order=3)
    filter_1.save_pkl(path, filename)
    filter_2 = ButterworthFilter.load_pkl(path, filename)
    assert(filter_1.get_params() == filter_2.get_params())
    assert(filter_1.get_params(deep=True) == filter_2.get_params(deep=True))


def test_store_load_filter_baseclass():
    path = os.path.join(Path(__file__).parent, "test_output")
    filename = "LPF.pickle"

    filter_1 = ButterworthFilter(T=20, order=3)
    filter_1.save_pkl(path, filename)

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    filter_3 = Filter.load_pkl(path, filename)
    assert(filter_2.get_params() == filter_3.get_params())
    assert (filter_2.get_params(deep=True) == filter_3.get_params(deep=True))


if __name__ == "__main__":
    test_filter_params_baseclass()
    test_store_load_filter()
    test_store_load_filter_baseclass()
