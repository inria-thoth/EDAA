import logging
import time

import numpy as np
import numpy.linalg as LA
from cvxopt import matrix, solvers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FCLS:
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    @staticmethod
    def _numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.vstack([A1, A2])

    @staticmethod
    def _numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    @staticmethod
    def _numpy_to_cvxopt_matrix(A):
        A = np.array(A, dtype=np.float64)
        if A.ndim == 1:
            return matrix(A, (A.shape[0], 1), "d")
        else:
            return matrix(A, A.shape, "d")

    def solve(self, Y, E):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U. Fully constrained least squares
        is least squares with the abundance sum-to-one constraint (ASC) and the
        abundance nonnegative constraint (ANC).

        Parameters:
            Y: `numpy array`
                2D data matrix (L x N).

            E: `numpy array`
                2D matrix of endmembers (L x p).

        Returns:
            X: `numpy array`
                2D abundance maps (p x N).

        References:
            Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
            Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.

        Notes:
            Three sources have been useful to build the algorithm:
                * The function hyperFclsMatlab, part of the Matlab Hyperspectral
                Toolbox of Isaac Gerg.
                * The Matlab (tm) help on lsqlin.
                * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
                http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
                , it's great code.
        """
        tic = time.time()
        assert len(Y.shape) == 2
        assert len(E.shape) == 2

        L1, N = Y.shape
        L2, p = E.shape

        assert L1 == L2

        # Reshape to match implementation
        M = np.copy(Y.T)
        U = np.copy(E.T)

        solvers.options["show_progress"] = False

        U = U.astype(np.double)

        C = self._numpy_to_cvxopt_matrix(U.T)
        Q = C.T * C

        lb_A = -np.eye(p)
        lb = np.repeat(0, p)
        A = self._numpy_None_vstack(None, lb_A)
        b = self._numpy_None_concatenate(None, -lb)
        A = self._numpy_to_cvxopt_matrix(A)
        b = self._numpy_to_cvxopt_matrix(b)

        Aeq = self._numpy_to_cvxopt_matrix(np.ones((1, p)))
        beq = self._numpy_to_cvxopt_matrix(np.ones(1))

        M = np.array(M, dtype=np.float64)
        M = M.astype(np.double)
        X = np.zeros((N, p), dtype=np.float32)
        for n1 in range(N):
            d = matrix(M[n1], (L1, 1), "d")
            q = -d.T * C
            sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)["x"]
            X[n1] = np.array(sol).squeeze()
        tac = time.time()
        logger.info(f"{self} took {tac - tic:.2f}s")
        return X.T


class FCLSv2:
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    @staticmethod
    def _numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.vstack([A1, A2])

    @staticmethod
    def _numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    @staticmethod
    def _numpy_to_cvxopt_matrix(A):
        A = np.array(A, dtype=np.float64)
        if A.ndim == 1:
            return matrix(A, (A.shape[0], 1), "d")
        else:
            return matrix(A, A.shape, "d")

    def solve(self, Y, p, seed=0, *args, **kwargs):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U. Fully constrained least squares
        is least squares with the abundance sum-to-one constraint (ASC) and the
        abundance nonnegative constraint (ANC).

        Parameters:
            Y: `numpy array`
                2D data matrix (L x N).

            E: `numpy array`
                2D matrix of endmembers (L x p).

        Returns:
            X: `numpy array`
                2D abundance maps (p x N).

        References:
            Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
            Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.

        Notes:
            Three sources have been useful to build the algorithm:
                * The function hyperFclsMatlab, part of the Matlab Hyperspectral
                Toolbox of Isaac Gerg.
                * The Matlab (tm) help on lsqlin.
                * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
                http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
                , it's great code.
        """
        tic = time.time()
        assert len(Y.shape) == 2

        L1, N = Y.shape

        E = VCA().solve(Y, p, seed=seed)

        assert len(E.shape) == 2
        L2, p = E.shape

        assert L1 == L2

        # Reshape to match implementation
        M = np.copy(Y.T)
        U = np.copy(E.T)

        solvers.options["show_progress"] = False

        U = U.astype(np.double)

        C = self._numpy_to_cvxopt_matrix(U.T)
        Q = C.T * C

        lb_A = -np.eye(p)
        lb = np.repeat(0, p)
        A = self._numpy_None_vstack(None, lb_A)
        b = self._numpy_None_concatenate(None, -lb)
        A = self._numpy_to_cvxopt_matrix(A)
        b = self._numpy_to_cvxopt_matrix(b)

        Aeq = self._numpy_to_cvxopt_matrix(np.ones((1, p)))
        beq = self._numpy_to_cvxopt_matrix(np.ones(1))

        M = np.array(M, dtype=np.float64)
        M = M.astype(np.double)
        X = np.zeros((N, p), dtype=np.float32)
        for n1 in range(N):
            d = matrix(M[n1], (L1, 1), "d")
            q = -d.T * C
            sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)["x"]
            X[n1] = np.array(sol).squeeze()
        tac = time.time()
        logger.info(f"{self} took {tac - tic:.2f}s")
        return E, X.T


class VCA:
    def __init__(self):
        pass

    def solve(self, Y, p, seed=0, snr_input=0):
        """
        Vertex Component Analysis

        ------- Input variables -------------
        HSI containing the following variables =>
         Y - matrix with dimensions L(channels) x N(pixels)
             each pixel is a linear mixture of R endmembers
             signatures Y = M x s, where s = gamma x alfa
             gamma is a illumination perturbation factor and
             alfa are the abundance fractions of each endmember.
         p - positive integer number of endmembers in the scene

        ------- Output variables -----------
        E     - estimated mixing matrix (endmembers signatures)

        ------- Optional parameters---------
        snr_input - (float) signal to noise ratio (dB)
        ------------------------------------

        Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        L, N = Y.shape
        self.seed = seed
        generator = np.random.RandomState(seed=seed)

        #############################################
        # SNR Estimates
        #############################################

        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                :, :p
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            logger.info(f"SNR estimated = {SNR:.2f}[dB]")
        else:
            SNR = snr_input
            logger.info(f"input SNR = {SNR:.2f}[dB]\n")

        SNR_th = 15 + 10 * np.log10(p)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            logger.info("... Select proj. to R-1")

            d = p - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            logger.info("... Select the projective proj.")

            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((p), dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1

        for i in range(p):
            w = generator.rand(p, 1)
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        logger.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape  # L number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est


if __name__ == "__main__":
    from hsi_unmixing.data.datasets.base import HSI

    hsi = HSI("JasperRidge.mat")
    hsi.plot_abundances(transpose=True)

    solver = FCLS()
    A0 = solver.solve(hsi.Y, hsi.E)
    hsi.plot_abundances(transpose=True, A0=A0)
