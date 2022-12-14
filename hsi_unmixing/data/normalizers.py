import logging
import pdb

import numpy as np
import numpy.linalg as LA
from hsi_unmixing import EPS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseNormalizer:
    def __init__(self, epsilon=EPS, dtype=np.float64):
        self.epsilon = epsilon
        self.dtype = dtype

    def transform(self, Y):
        raise NotImplementedError


class RawInput(BaseNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, Y):
        return Y.astype(self.dtype)


class GlobalMinMax(BaseNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, Y):
        num = Y - Y.min()
        denom = (Y.max() - Y.min()) + self.epsilon
        return (num / denom).astype(self.dtype)


class BandwiseMinMax(BaseNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, Y):
        minValuePerBand = Y.min(axis=1, keepdims=True)
        maxValuePerBand = Y.max(axis=1, keepdims=True)
        num = Y - minValuePerBand
        denom = (maxValuePerBand - minValuePerBand) + self.epsilon
        return (num / denom).astype(self.dtype)


class PixelwiseNorm(BaseNormalizer):
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def transform(self, Y):
        assert len(Y.shape) == 2
        num = Y
        denom = LA.norm(Y, axis=0, ord=self.order, keepdims=True)
        return (num / denom).astype(self.dtype)


class PixelwiseL1Norm(PixelwiseNorm):
    def __init__(self):
        super().__init__(order=1)


class PixelwiseL2Norm(PixelwiseNorm):
    def __init__(self):
        super().__init__(order=2)


class PixelwiseNormCutoff(BaseNormalizer):
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def transform(self, Y):
        assert len(Y.shape) == 2
        num = Y
        norm = LA.norm(Y, axis=0, ord=self.order, keepdims=True)
        cutoff = 0.25 * norm.mean()
        logger.info(f"Cutoff => {cutoff}")
        denom = np.maximum(norm, cutoff)
        res = (num / denom).astype(self.dtype)
        return res


class PixelwiseL1NormCutoff(PixelwiseNormCutoff):
    def __init__(self):
        super().__init__(order=1)


class PixelwiseL2NormCutoff(PixelwiseNormCutoff):
    def __init__(self):
        super().__init__(order=2)
