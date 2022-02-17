import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class APEX4Dataset(HSI):
    def __init__(self, **kwargs):
        # super().__init__("APEX4.mat", **kwargs)
        super().__init__("APEX4fixed.mat", **kwargs)


class APEX6Dataset(HSI):
    def __init__(self, **kwargs):
        # super().__init__("APEX6.mat", **kwargs)
        super().__init__("APEX6fixed.mat", **kwargs)


class APEX4OldDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("APEX4.mat", **kwargs)


class APEX6OldDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("APEX6.mat", **kwargs)


class TinyAPEXDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("TinyAPEX.mat", **kwargs)


if __name__ == "__main__":
    from hsi_unmixing.data.normalizers import GlobalMinMax as GMM
    # from hsi_unmixing.data.normalizers import PixelwiseL1Norm as PL1
    from hsi_unmixing.models.supervised import DecompSimplex as DS

    setter = DS()
    normalizer = GMM()
    # normalizer = PL1()
    hsi = APEX4Dataset(
        normalizer=normalizer,
        setter=setter,
    )
    print(hsi)
    hsi.plot_endmembers(normalize=False)
    # hsi.plot_hsi(SNR=10)
    # hsi.plot_abundances(grid=(2, 3))

    hsi.plot_abundances()
