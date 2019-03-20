import numpy as np
from time import time

import matlab.engine

eng = matlab.engine.start_matlab()


# TODO: Add a smoothing term
def update_x(W, Hom, x, layers, depth, mu):
    print('Updating x...')
    start = time()

    h, w, c = x.shape
    t = Hom.shape[1]
    N = len(layers)

    new_x = np.zeros(x.shape)
    for i in range(N):
        x_i = np.zeros(x.shape)
        x_y, x_x = np.where(layers[i] == 1)
        x_i[x_y, x_x] = x[x_y, x_x]
        x_i = x_i.reshape((h, -1))

        kernel = np.matmul(np.nan_to_num(W[np.newaxis, i]), Hom[i].reshape(t, -1)).reshape(3, 3)
        # D. Krishnan, R. Fergus:
        # "Fast Image Deconvolution using Hyper-Laplacian Priors"
        s = time()
        x_mat = matlab.double(x_i.tolist())
        k_mat = matlab.double(kernel.tolist())
        print('converting to mat Delt(time): %.2f' % (time() - s))
        output = eng.fast_deconv(x_mat, k_mat, mu, 0.5, nargout=1)
        new_x += np.array(output).reshape((h, w, c))

    print('Updated x. Delt(Time): %.2f' % (time() - start))
    return new_x
