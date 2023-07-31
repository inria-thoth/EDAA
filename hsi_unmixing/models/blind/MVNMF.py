import logging
import os
import time
import warnings

import numpy as np

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. minvolNMF will not work")

VALID_PATHS = [
    # "/home/clear/azouaoui/code/matlab/NMF-QMV_demo",
    "/home/azouaoui/CH4/NCAA_v1/utils"
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class minvolNMF:
    """ """

    def __init__(
        self,
        path_to_NMFQMV: str = VALID_PATHS[0],
    ):
        assert os.path.exists(path_to_NMFQMV), "Change path to your location of NMFQMV"

        # Start matlab engine
        self.eng = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to NMF-QMV code
        self.eng.cd(path_to_NMFQMV)

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(self, Y, p, *args, **kwargs):
        tic = time.time()

        L, N = Y.shape
        options = {"datatype": "real", "maxiter": 200}
        Ehat, Ahat = self.eng.minvolNMF(
            matlab.double(Y.tolist()),
            matlab.int64([p]),
            options,
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
