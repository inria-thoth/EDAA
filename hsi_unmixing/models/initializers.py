import logging
import pdb

import numpy as np
import numpy.linalg as LA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseInit:
    def __init__(self):
        self.seed = None

    def init_like(self, hsi):
        return NotImplementedError

    def __repr__(self):
        msg = "f{self.__class__.__name__}_seed{self.seed}"
        return msg


class TrueEndmembers(BaseInit):
    def __init__(self):
        super().__init__()

    def init_like(self, hsi):
        return hsi.E


class RandomPositiveMatrix(BaseInit):
    def __init__(self):
        super().__init__()

    def init_like(self, hsi, seed=0):
        self.seed = seed
        generator = np.random.RandomState(seed=self.seed)
        return generator.rand(hsi.L, hsi.p)


class RandomPixels(BaseInit):
    def __init__(self):
        super().__init__()

    def init_like(self, hsi, seed=0):
        self.seed = seed
        generator = np.random.RandomState(seed=self.seed)
        indices = generator.randint(0, high=hsi.N - 1, size=hsi.p)
        logger.debug(f"Indices randomly chosen: {indices}")
        pixels = hsi.Y[:, indices]
        assert pixels.shape == hsi.E.shape
        return pixels


class VCA(BaseInit):
    def __init__(self):
        super().__init__()

    def init_like(self, hsi, seed=0, snr_input=0):
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
        Y = hsi.Y
        N, p = hsi.N, hsi.p
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

            logger.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            logger.info(f"input SNR = {SNR}[dB]\n")

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
            c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
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

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape  # L number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y ** 2) / float(N)
        P_x = np.sum(x ** 2) / float(N) + np.sum(r_m ** 2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est


if __name__ == "__main__":
    from hsi_unmixing.data.datasets.base import HSI

    hsi = HSI("Samson.mat")

    # hsi.plot_endmembers()

    # initializer = TrueEndmembers()
    # initializer = RandomPositiveMatrix()
    # initializer = RandomPixels()
    initializer = VCA()
    E = initializer.init_like(hsi)

    hsi.plot_endmembers(E0=E)
