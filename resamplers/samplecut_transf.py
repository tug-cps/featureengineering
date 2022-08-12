from ..interfaces import BasicTransformer


class SampleCut_Transf(BasicTransformer):
    num_samples = 0

    def __init__(self, num_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def transform(self, X):
        return X[self.num_samples:]