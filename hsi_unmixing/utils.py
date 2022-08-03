import logging

import scipy.io as sio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_estimates(Ehat, Ahat, hsi):
    data = {
        "Ehat": Ehat,
        "Egt": hsi.scaledE,
        "Ahat": Ahat,
        "Agt": hsi.A,
        "H": hsi.H,
        "W": hsi.W,
    }
    sio.savemat("estimates.mat", data)


def load_estimates(path: str):
    data = sio.loadmat(path)
    return data
