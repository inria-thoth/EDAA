import logging
import os
import time
import warnings

import numpy as np

from hsi_unmixing.models.blind.ADMMNet import VCA

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. SeCoDe will not work")

VALID_PATHS = ["/home/azouaoui/CH4/IEEE_TGRS_SeCoDe"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SeCoDe:
    """
    SeCoDe
    """

    def __init__(
        self,
        path_to_SeCoDe: str = VALID_PATHS[0],
    ):
        assert os.path.exists(path_to_SeCoDe), "Change path to your location of NCAA"

        # Start matlab engine
        self.eng = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to NCAA code
        self.eng.cd(path_to_SeCoDe)

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(self, Y, p, H, W, *args, **kwargs):
        tic = time.time()

        breakpoint()
        E0 = VCA().extract_endmembers(Y, p)

        Ehat, Ahat = self.eng.SeCoDe(
            matlab.double(Y.tolist()),
            matlab.int64([p]),
            matlab.double(E0.tolist()),
            matlab.int64([H]),
            matlab.int64([W]),
            nargout=2,
        )

        Ehat = np.array(Ehat).astype(np.float32)
        Ahat = np.array(Ahat).astype(np.float32)

        self.E = Ehat
        self.A = Ahat

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"{self} took {self.time}s")

        return self.E, self.A
