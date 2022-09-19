import logging

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TinyAPEXDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("TinyAPEX.mat", **kwargs)


if __name__ == "__main__":
    from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
    from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

    normalizer = BMM()
    # normalizer = PL2()

    hsi = TinyAPEXDataset(normalizer=normalizer)
    print(hsi)
    hsi.plot_hsi(channels=[200, 100, 10], sort_channels=False)
    # hsi.plot_endmembers()
