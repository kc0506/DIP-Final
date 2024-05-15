import os
from typing import Literal

import cv2
import numpy as np

# --------------------------------- Image I/O -------------------------------- #


def read_img(path: str) -> np.ndarray:
    return cv2.imread(path)


def write_img(path: str | os.PathLike, img: np.ndarray):
    if img.max() <= 10:
        img = img.astype(bool)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    # if img.dtype == np.complex128:
    #     img = np.abs(img).astype(np.uint8)

    img = np.clip(img, 0, 255)
    path = str(path)
    if not path.endswith(".png"):
        path += ".png"
    return cv2.imwrite(path, img)


# ------------------------------- Basic Utility ------------------------------ #


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img

    n, m, c = img.shape
    if c != 3:
        return img

    fac = np.array([0.299, 0.587, 0.114])
    fac = fac / fac.sum()  # normalize
    return np.dot(img, fac).astype(np.uint8)


def gray_to_rgb(img: np.ndarray):
    return img[..., None].repeat(3, axis=-1)


def gray_to_channel(
    img: np.ndarray,
    channel: Literal["b", "g", "r"],
) -> np.ndarray:
    if img.dtype == bool or img.max() <= 10:
        img = img.astype(np.uint8) * 255

    idx = "bgr".index(channel)
    res = np.zeros(img.shape + (3,))
    res[..., idx] = img
    return res


def is_bool(img: np.ndarray) -> bool:
    return len(img.shape) == 2 and img.dtype == bool


def is_gray(img: np.ndarray) -> bool:
    return len(img.shape) == 2 and img.dtype == np.uint8


def is_color(img: np.ndarray) -> bool:
    return len(img.shape) == 3 and img.dtype == np.uint8
