import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Urban4Dataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Urban4.mat", **kwargs)


class Urban5Dataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Urban5.mat", **kwargs)


class Urban6Dataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Urban6.mat", **kwargs)


class Urban4RadiusDataset(HSI):
    def __init__(self, radius, **kwargs):
        super().__init__(f"Urban4_r{radius}.mat", **kwargs)


if __name__ == "__main__":
    hsi4 = Urban4Dataset()
    print(hsi4)
    # hsi4.plot_endmembers(normalize=False)
    # hsi4.plot_abundances(transpose=True)
    hsi5 = Urban5Dataset()
    print(hsi5)
    # hsi5.plot_endmembers(normalize=False)
    hsi6 = Urban6Dataset()
    print(hsi6)
    hsi6.plot_endmembers(normalize=True)
    hsi6.plot_endmembers(normalize=False)
    # hsi6.plot_abundances(transpose=True)
