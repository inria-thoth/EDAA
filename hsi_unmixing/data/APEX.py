import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class APEX4Dataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("APEX4.mat", data_path)


if __name__ == "__main__":
    hsi = APEX4Dataset()
    print(hsi)
