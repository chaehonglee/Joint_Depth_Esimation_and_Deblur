# Code referenced from WUSTL CSE559 assignment 1
import numpy as np


def bfilt(im, k, sgm_s, sgm_i):
    """
    bilateral filter
    :param im: input image
    :param k: kernel size
    :param sgm_s: spacial variance
    :param sgm_i: intensity variance
    :return: filtered image
    """
    h, w, c = im.shape
    yy = np.zeros(im.shape)
    b = np.zeros([h, w, 1])

    for y in range(-k, k + 1):
        for x in range(-k, k + 1):
            if y < 0:
                y1a = 0
                y1b = -y
                y2a = h + y
                y2b = h
            else:
                y1a = y
                y1b = 0
                y2a = h
                y2b = h - y

            if x < 0:
                x1a = 0
                x1b = -x
                x2a = w + x
                x2b = w
            else:
                x1a = x
                x1b = 0
                x2a = w
                x2b = w - x

            bxy = im[y1a:y2a, x1a:x2a, :] - im[y1b:y2b, x1b:x2b, :]
            bxy = np.sum(bxy * bxy, axis=2, keepdims=True)

            bxy = bxy / (sgm_i ** 2) + np.float32(y ** 2 + x ** 2) / (sgm_s ** 2)
            bxy = np.exp(-bxy / 2.0)

            b[y1b:y2b, x1b:x2b, :] += bxy
            yy[y1b:y2b, x1b:x2b, :] += bxy * im[y1a:y2a, x1a:x2a, :]

    return yy / b
