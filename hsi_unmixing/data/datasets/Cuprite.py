import logging
import pdb

from .base import HSI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CupriteDataset(HSI):
    def __init__(self, **kwargs):
        super().__init__("Cuprite.mat", **kwargs)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")

    hsi = CupriteDataset()
    print(hsi)
    # hsi.plot_endmembers()
    hsi.plot_abundances(grid=(3, 4))
