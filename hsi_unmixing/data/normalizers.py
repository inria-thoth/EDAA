import logging
import pdb

import numpy as np
import numpy.linalg as LA
from hsi_unmixing import EPS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RawInput:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def transform(self, Y):
        return Y.astype(self.dtype)


class GlobalMinMax:
    def __init__(self, epsilon=EPS):
        self.epsilon = epsilon

    def transform(self, Y):
        num = Y - Y.min()
        denom = (Y.max() - Y.min()) + self.epsilon
        return num / denom


class BandwiseMinMax:
    def __init__(self, epsilon=EPS):
        self.epsilon = epsilon

    def transform(self, Y):
        minValuePerBand = Y.min(axis=1, keepdims=True)
        maxValuePerBand = Y.max(axis=1, keepdims=True)
        num = Y - minValuePerBand
        denom = (maxValuePerBand - minValuePerBand) + self.epsilon
        return num / denom


class PixelwiseNorm:
    def __init__(self, order):
        self.order = order

    def transform(self, Y):
        assert len(Y.shape) == 2
        # Expect L (# HSI channels) x N (# HSI pixels)
        assert Y.shape[0] < Y.shape[1]
        return Y / LA.norm(Y, axis=1, ord=self.order, keepdims=True)


class PixelwiseL1Norm(PixelwiseNorm):
    def __init__(self):
        super().__init__(order=1)


class PixelwiseL2Norm(PixelwiseNorm):
    def __init__(self):
        super().__init__(order=2)
