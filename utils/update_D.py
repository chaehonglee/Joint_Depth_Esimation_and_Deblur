import time
import numpy as np
from utils.compute_K import compute_KX


# 1. Apply L0 smoothing on x (but do not alter x)
# 2. Discard small values of alpha_i,k
# 3. assign alpha_i,k to x_i and y_i
def update_D(y, x, W, Hom, layers, V, depth, eta):
    """
    Updates the depth values according to expectation
    :param y: blurry image
    :param x: predicted image at this iteration
    :param W: predicted exposure weights
    :param Hom: Homography warping matrices
    :param layers: matted layers
    :param V: possible values of depths
    :param depth: currently predicted depth
    :param eta: eta
    :return: returns the next predicted depth
    """
    print('Updating D...')
    start = time()

    h, w, c = x.shape
    s, t = Hom.shape[0], Hom.shape[1]
    N = len(layers)

    for i in range(N):
        x_i = np.zeros(x.shape)
        y_i = np.zeros(y.shape)
        x_y, x_x = np.where(layers[i] == 1)
        x_i[x_y, x_x] = x[x_y, x_x]
        y_i[x_y, x_x] = y[x_y, x_x]

        d_i = x_i[x_y[0], x_x[0], 0]
        idx = (np.abs(V - d_i)).argmin()

        kx = compute_KX(Hom[idx], x_i)
        kx = kx.reshape((t, -1))
        wkx = np.matmul(W[i, np.newaxis, :], kx)
        wkx = wkx.reshape((h, w, c))

        dist = np.linalg.norm(y_i - wkx) ** 2
        numerator = np.exp(-dist / (2 * eta ** 2))

        denominator = 0
        for k in range(s):
            kx = compute_KX(Hom[k], x_i)
            kx = kx.reshape((t, -1))
            wkx = np.matmul(W[i, np.newaxis, :], kx)
            wkx = wkx.reshape((h, w, c))

            dist = np.linalg.norm(y_i - wkx) ** 2
            denominator += np.exp(-dist / (2 * eta ** 2))

        depth[x_y, x_x] = np.nan_to_num(numerator / denominator)

    print('Updated D. Delt(Time): %.2f' % (time() - start))
    return depth
