import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimulatedDataCubesNoPurePixelsDataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("SimulatedDataCubesNoPurePixels.mat", data_path)


class SimulatedDataCubesNoPurePixelsHardDataset(HSI):
    def __init__(self, data_path: str = "./data"):
        super().__init__("SimulatedDataCubesNoPurePixelsHard.mat", data_path)


if __name__ == "__main__":
    hsi1 = SimulatedDataCubesNoPurePixelsDataset()
    print(hsi1)
    # hsi1.plot_endmembers()
    # hsi1.plot_abundances()
    for snr in [20, 30, 40, 50]:
        hsi1.plot_hsi(SNR=snr)
    hsi2 = SimulatedDataCubesNoPurePixelsHardDataset()
    print(hsi2)
