import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SamsonDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Samson.mat", **kwargs)


if __name__ == "__main__":
    hsi = SamsonDataset()
    print(hsi)
