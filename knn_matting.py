# Code referenced from https://github.com/MarcoForte/knn-matting
import numpy as np
import sklearn.neighbors
import scipy.sparse
import scipy.misc
import warnings

nn = 10


def knn_matte(img, trimap, mylambda=100):
    """
    Computes matting layer from input image and scribble trimap
    :param img: input image
    :param trimap: Scribble indicator
    :param mylambda:
    :return: matted layer
    """
    [m, n, c] = img.shape
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background

    print('Finding nearest neighbors')
    a, b = np.unravel_index(np.arange(m * n), (m, n))
    feature_vec = np.append(np.transpose(img.reshape(m * n, c)), [a, b] / np.sqrt(m * m + n * n), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print('Computing sparse A')
    row_inds = np.repeat(np.arange(m * n), 10)
    col_inds = knns.reshape(m * n * 10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1) / (c + 2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(m * n, m * n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script - A
    D = scipy.sparse.diags(np.ravel(all_constraints))
    v = np.ravel(foreground)
    c = 2 * mylambda * np.transpose(v)
    H = 2 * (L + mylambda * D)

    print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)
    return alpha