import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class JasperRidgeDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("JasperRidge.mat", **kwargs)


class JasperRidgeRadiusDataset(HSI):
    def __init__(self, radius, **kwargs):
        super().__init__(f"JasperRidge_r{radius}.mat", **kwargs)


if __name__ == "__main__":
    hsi = JasperRidgeDataset()
    print(hsi)
