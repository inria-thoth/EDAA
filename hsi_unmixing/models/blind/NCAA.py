import logging
import os
import time
import warnings

import numpy as np

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. NCAA will not work")

VALID_PATHS = ["/home/azouaoui/CH4/NCAA_v1"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NCAA:
    """
    Nearly Convex AA
    """

    def __init__(
        self,
        path_to_NCAA: str = VALID_PATHS[0],
        n_clust=20,
    ):
        assert os.path.exists(path_to_NCAA), "Change path to your location of NCAA"

        self.n_clust = n_clust

        # Start matlab engine
        self.eng = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to NCAA code
        self.eng.cd(path_to_NCAA)

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(self, Y, p, *args, **kwargs):
        tic = time.time()

        options = {"fine_tuning": 1}
        Bhat, Ahat, Yhat = self.eng.NCAA(
            matlab.double(Y.tolist()),
            matlab.int64([p]),
            matlab.int64([self.n_clust]),
            options,
            nargout=3,
        )

        Yhat = np.array(Yhat).astype(np.float32)
        Bhat = np.array(Bhat).astype(np.float32)
        Ahat = np.array(Ahat).astype(np.float32)

        self.E = Yhat @ Bhat
        self.A = Ahat

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"{self} took {self.time}s")

        return self.E, self.A
