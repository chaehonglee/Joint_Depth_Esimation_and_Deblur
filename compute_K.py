import numpy as np


def getHomography(im, fl, theta, trans, numP, v_k):
    """
    Generates possible Homography
    :param im: image
    :param fl: focal lenth
    :param theta: list of possible rotation angle
    :param trans: list of possible translation
    :param numP: number of possible camera pose T
    :param v_k: constant depth v_k
    :return: set of possible Homorgraphy P
    """
    h, w, c = im.shape

    # 3 x 3 intrinsic matrix
    C = np.array([[fl, 0, w / 2], [fl, h / 2, 0], [0, 0, 1]])
    C_inv = np.linalg.inv(C)

    # Rotational matrix
    Rcos = np.cos(theta);
    Rsin = np.sin(theta)
    Rx = np.zeros((numP, 3, 3))
    Rz = np.zeros((numP, 3, 3))
    Ry = np.zeros((numP, 3, 3))
    Rx[:, 0, 0], Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = np.ones(numP), Rcos, Rsin, -Rsin, Rcos
    Ry[:, 1, 1], Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 0, 2] = np.ones(numP), Rcos, Rsin, -Rsin, Rcos
    Rz[:, -1, -1], Rz[:, 0, 0], Rz[:, 1, 0], Rz[:, 0, 1], Rz[:, 1, 1] = np.ones(numP), Rcos, Rsin, -Rsin, Rcos
    R = np.matmul(np.matmul(Rx, Ry), Rz)

    # Translational matrix
    T = np.zeros((numP, 3, 3))
    T[:, -1, 0] = trans
    T[:, -1, 1] = trans
    T[:, -1, 2] = np.ones(numP)

    P = np.zeros([numP, 3, 3])
    for j in range(numP):
        RT = np.add(R[j], T[j] / v_k)
        P[j, :, :] = np.matmul(np.matmul(C, RT), C_inv)

    return P


# Code partially referenced from WUSTL CSE559 assignment 3
def compute_KX(Hom, im):
    """
    Computes transformation on an image with the given Homography
    :param Hom: list of Homography matrices
    :param im: input im
    :return: transformed x by the homography
    """
    h, w, c = im.shape
    t, _, _ = Hom.shape
    KX = np.zeros((t, h, w, c))

    for i in range(t):
        H = Hom[i]

        y, x = np.meshgrid(range(h), range(w), indexing='ij')
        y, x = np.reshape(y, [-1, 1]), np.reshape(x, [-1, 1])

        xyd = np.concatenate([x, y], axis=1)
        xydH = np.concatenate([xyd, np.ones((x.shape[0], 1))], axis=1)
        xysH = np.matmul(H, xydH.T).T
        xys = xysH[:, 0:2] / xysH[:, 2:3]

        cnd = np.logical_and(xys[:, 0] > 0, xys[:, 1] > 0)
        cnd = np.logical_and(cnd, xys[:, 0] < w - 1)
        cnd = np.logical_and(cnd, xys[:, 1] < h - 1)
        idx = np.where(cnd)[0]
        xyd = np.int64(xyd[idx, :])
        xys = xys[idx, :]

        xysf = np.int64(np.floor(xys))
        xysc = np.int64(np.ceil(xys))
        xalph = xys[:, 0:1] - np.floor(xys[:, 0:1])
        yalph = xys[:, 1:2] - np.floor(xys[:, 1:2])
        xff = im[xysf[:, 1], xysf[:, 0], :]
        xfc = im[xysf[:, 1], xysc[:, 0], :]
        xcf = im[xysc[:, 1], xysf[:, 0], :]
        xcc = im[xysc[:, 1], xysc[:, 0], :]

        KX[i, xyd[:, 1], xyd[:, 0], :] = (1 - xalph) * ((1 - yalph) * xff + yalph * xcf) \
                                         + xalph * ((1 - yalph) * xfc + yalph * xcc)
    return KX
