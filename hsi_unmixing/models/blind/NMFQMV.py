import logging
import os
import pdb
import time
import warnings

import numpy as np

try:
    import matlab.engine
except Exception:
    warnings.warn("matlab.engine was not imported. NMF-QMV will not work")

VALID_PATHS = [
    "/home/azouaoui/matlab/NMF-QMV_demo",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NMFQMV:
    def __init__(
        self,
        beta_candidates=np.logspace(-2, 2, 5),
        term: str = "boundary",
        path_to_NMFQMV: str = VALID_PATHS[0],
        drawfigs: str = "no",
    ):
        assert os.path.exists(path_to_NMFQMV)

        self.term = term
        self.betas = beta_candidates
        self.drawfigs = drawfigs

        # Start matlab engine
        self.eng = matlab.engine.start_matlab()
        logger.debug("MATLAB engine started")
        # Go to NMF-QMV code
        self.eng.cd(path_to_NMFQMV)

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(self, img, p, H, W, *args, **kwargs):
        tic = time.time()

        L, N = img.shape
        img = img.T.reshape(H, W, L)
        img = img.transpose(1, 0, 2)
        _, Ehat, Ahat = self.eng.NMF_QMV(
            matlab.double(img.tolist()),
            matlab.double([p]),
            matlab.double(self.betas.tolist()),
            self.term,
            "DRAWFIGS",
            self.drawfigs,
            nargout=3,
        )

        Ehat = np.array(Ehat).astype(np.float32)
        Ahat = np.array(Ahat).astype(np.float32)

        self.E = Ehat
        self.A = Ahat

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"{self} took {self.time}s")

        return self.E, self.A


if __name__ == "__main__":
    L, H, W = 32, 16, 16
    p = 3

    img = np.random.rand(L * H * W).reshape(H, W, L)

    solver = NMFQMV()

    solver.solve(img, p)
