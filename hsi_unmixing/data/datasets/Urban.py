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
    import matplotlib
    import matplotlib.pyplot as plt
    from hsi_unmixing.data.normalizers import BandwiseMinMax as BMM
    from hsi_unmixing.data.normalizers import GlobalMinMax as GMM
    from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2

    matplotlib.use("TKAgg")

    # plt.use("TKAgg")

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
    # hsi.plot_endmembers()
    # hsi5 = Urban5Dataset(normalizer=normalizer)
    # print(hsi5)
    # hsi4.plot_endmembers(normalize=False)
    # hsi4.plot_abundances(transpose=True)
    # hsi5 = Urban5Dataset()
    # print(hsi5)
    # # hsi5.plot_endmembers(normalize=False)
    # hsi6 = Urban6Dataset()
    # print(hsi6)
    # hsi6.plot_endmembers(normalize=True)
    # hsi6.plot_endmembers(normalize=False)
    # # hsi6.plot_abundances(transpose=True)
