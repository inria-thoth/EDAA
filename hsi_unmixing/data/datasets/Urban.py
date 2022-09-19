import logging

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


if __name__ == "__main__":
    from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
    from hsi_unmixing.data.normalizers import GlobalMinMax as GMM
    from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

    # normalizer = PL2()
    # normalizer = GMM()
    normalizer = BMM()
    hsi = Urban6Dataset(normalizer=normalizer)
    print(hsi)
    hsi.plot_hsi(
        channels=[130, 70, 30],
        sort_channels=False,
        transpose=True,
    )
