import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class JasperRidgeDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("JasperRidge.mat", **kwargs)


if __name__ == "__main__":
    hsi = JasperRidgeDataset()
    print(hsi)
