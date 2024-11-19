import cv2
import numpy as np


def crop_center_square(img: np.ndarray):
    h, w = img.shape[:2]

    if h != w:
        offset = (w - h) // 2
        img = img[:, offset:-offset]

    # img = cv2.resize(img, (512, 512))
    # p = 100
    # img = img[p:-p, p:-p]

    return img
