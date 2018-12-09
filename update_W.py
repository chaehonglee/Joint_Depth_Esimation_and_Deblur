from time import time
import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient

from project.compute_K import compute_KX


# TODO: Add a smoothing term
def update_W(W, Hom, x, layers, depth, lmbda):
    """
    Updates W = {w_1, ..., w_N} using conjugate gradient method
    :param w:
    :param Hom: s x t x 3 x 3
    :param x:
    :param layers:
    :param depth:
    :param lmbda:
    :return:
    """
    print('Updating W...')
    start = time()

    h, w, c = x.shape
    t = Hom.shape[1]
    N = len(layers)

    # build matrix A, b, AtA, Atb:
    A = np.zeros((N, t, h, w, c))
    b = np.zeros((N, h, w, c))
    # gamma = np.zeros((N, h, w, c))

    Atb = np.zeros((N, t))
    AtA = np.zeros((N, t, t))

    for i in range(N):
        x_i = np.zeros(x.shape)
        x_y, x_x = np.where(layers[i] == 1)
        x_i[x_y, x_x] = x[x_y, x_x]

        d_i = x_i[x_y[0], x_x[0], 0]
        idx = (np.abs(depth - d_i)).argmin()

        A[i] = compute_KX(Hom[idx], x_i)
        b[i, 1:] = np.diff(x_i, axis=0)
        # gamma[i, 1:] = np.diff(x_i, axis=0)
        # gamma[i, :, 1:] += np.diff(x_i, axis=1)

        Atb[i] = np.matmul(A[i].reshape((t, -1)), b[i].reshape(-1))
        AtA[i] = np.matmul(A[i].reshape((t, -1)), A[i].reshape((t, -1)).T)
        W[i] = conjugate_gradient(AtA[i], Atb[i])[0]
        W[i] /= np.sum(W[i])

    print('Updated W. Delt(Time): %.2f' % (time() - start))
    return W
