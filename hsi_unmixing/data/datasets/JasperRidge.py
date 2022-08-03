import logging

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class JasperRidgeDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("JasperRidge.mat", **kwargs)


class JasperRidgeRadiusDataset(HSI):
    def __init__(self, radius, **kwargs):
        super().__init__(f"JasperRidge_r{radius}.mat", **kwargs)


if __name__ == "__main__":

    from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
    from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

    normalizer = BMM()
    # normalizer = PL2()

    hsi = JasperRidgeDataset(normalizer=normalizer)
    print(hsi)
    hsi.plot_hsi(
        channels=[80, 50, 10],
        sort_channels=False,
        transpose=True,
    )
    # hsi.plot_endmembers()
