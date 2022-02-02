import logging
import pdb

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AdditiveWhiteGaussianNoise:
    def __init__(self):
        self.sigmas = None
        self.SNR = None
        self.L = None
        self.N = None

    def fit(self, Y, SNR: float):
        """
        Compute sigmas at the desired SNR given a flattened input HSI Y
        """
        assert len(Y.shape) == 2
        self.L, self.N = Y.shape
        logger.debug(f"Y shape: {Y.shape}")
        self.SNR = SNR
        logger.debug(f"Desired SNR: {self.SNR}")

        if SNR is None:
            self.sigmas = np.zeros(self.L)
        else:
            assert SNR > 0, "SNR must be stricly positive"
            # Uniform across bands
            sigmas = np.ones(self.L)
            # Normalization
            sigmas /= np.linalg.norm(sigmas)
            logger.debug(f"Sigmas after normalization: {sigmas}")
            # compute mean sigma
            num = np.sum(Y ** 2) / self.N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            logger.debug(f"Sigma mean based on SNR: {sigmas_mean}")
            # Noise variance
            sigmas *= sigmas_mean
            logger.debug(f"Final sigmas value: {sigmas}")
            self.sigmas = sigmas

    def transform(self, Y, seed=0):
        """
        Add White Gaussian Noise to the flattened input HSI Y
        """
        assert self.sigmas is not None, "Must be fitted first"
        assert (self.L, self.N) == Y.shape
        # Fix random seed
        generator = np.random.RandomState(seed=seed)
        # Noise generation
        N = np.diag(self.sigmas) @ generator.randn(self.L, self.N)
        # Additive Noise
        return Y + N

    def fit_transform(self, Y, SNR, seed=0):
        """
        Combine `fit` and `transform` methods.
        See above for their respective usage.
        """
        self.fit(Y, SNR)
        return self.transform(Y, seed=seed)
