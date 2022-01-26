import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CupriteDataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("Cuprite.mat", data_path)


if __name__ == "__main__":
    hsi = CupriteDataset()
    print(hsi)
