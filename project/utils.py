import os
from typing import Literal

import cv2
import numpy as np

# --------------------------------- Image I/O -------------------------------- #


def read_img(path: str) -> np.ndarray:
    return cv2.imread(path).astype(np.uint8)


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


def rgb2gray(img: np.ndarray) -> np.ndarray:
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
    return len(img.shape) == 3


def rgb2hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def remap(img: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.interp(img, (img.min(), img.max()), (low, high)).astype(np.uint8)


# ------------------------ Structuring Element Utility ----------------------- #


def get_layers(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    se = se.astype(bool)
    se_size = se.sum()
    cx, cy = se.shape[0] // 2, se.shape[1] // 2

    x, y = np.nonzero(se)
    shift_x = x - cx
    shift_y = y - cy
    xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
    xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, img.shape[0] - 1)
    ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, img.shape[1] - 1)

    return img[xs, ys]
