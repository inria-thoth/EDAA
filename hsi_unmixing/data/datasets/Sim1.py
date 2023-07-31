import logging

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Sim1Dataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Sim1.mat", **kwargs)


if __name__ == "__main__":
    from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
    from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

    normalizer = PL2()
    # normalizer = BMM()

    hsi = Sim1Dataset(normalizer=normalizer)
    print(hsi)
    # hsi.plot_hsi(
    #     channels=[83, 43, 9],
    #     sort_channels=False,
    #     transpose=True,
    # )
    hsi.plot_endmembers()
