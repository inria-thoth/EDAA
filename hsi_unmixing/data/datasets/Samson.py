import logging
import pdb

from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SamsonDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Samson.mat", **kwargs)


class SamsonFixedDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("SamsonFixed.mat", **kwargs)


if __name__ == "__main__":

    normalizer = PL2()
    # normalizer = BMM()
    # normalizer = None

    hsi = SamsonDataset(normalizer=normalizer)
    print(hsi)
    # hsi.plot_hsi(
    #     channels=[83, 43, 9],
    #     sort_channels=False,
    #     transpose=True,
    # )
    hsi.plot_endmembers()
