import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class APEX4Dataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("APEX4.mat", data_path)


class APEX6Dataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("APEX6.mat", data_path)


if __name__ == "__main__":
    hsi = APEX4Dataset()
    print(hsi)
    # hsi.plot_endmembers(normalize=True)
    hsi.plot_hsi(SNR=10)
    # hsi.plot_abundances(grid=(2, 3))
