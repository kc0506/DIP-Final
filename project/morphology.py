import atexit
import re
from hashlib import sha1
from time import time
from turtle import shape
from typing import Callable, Literal

import cv2
import numpy as np
from order import BasicOrder, ColorOrder, RGBOrder, get_default_order
from sample import Sample, get_sample
from task import write_ref
from utils import is_bool, is_color, is_gray, write_img


class SE:
    """
    Structuring element
    """

    @staticmethod
    def circle(r: int, thickness: int = -1) -> np.ndarray:
        d = 2 * r
        img = np.zeros((d + 1, d + 1), dtype=np.uint8)
        cv2.circle(img, (r, r), r, 255.0, thickness=thickness)  # type: ignore
        return img

    @staticmethod
    def square(r: int) -> np.ndarray:
        return np.full((r, r), 1)

    @staticmethod
    def cross(r: int) -> np.ndarray:
        img = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        cv2.line(img, (r, 0), (r, 2 * r), 255.0)  # type: ignore
        cv2.line(img, (0, r), (2 * r, r), 255.0)  # type: ignore
        return img


# ---------------------------------- Utility --------------------------------- #


def pad(img: np.ndarray, kernel_shape: tuple[int, ...]) -> np.ndarray:
    l_x = (kernel_shape[0] - 1) // 2
    l_y = (kernel_shape[1] - 1) // 2

    # return np.pad(img, ((l_x, l_x), (l_y,l_y)), mode='constant', constant_values=0)
    return np.pad(img, ((l_x, l_x), (l_y, l_y)), mode="edge")


def get_submat(arr: np.ndarray, shape: tuple[int, ...], to_pad=True) -> np.ndarray:
    """
    Ref: https://stackoverflow.com/questions/19414673/in-numpy-how-to-efficiently-list-all-fixed-size-submatrices
    """

    assert len(arr.shape) == 2
    assert len(shape) == 2

    if to_pad:
        arr = pad(arr, shape)

    view_shape = tuple(np.subtract(arr.shape, shape) + 1) + shape

    # ? stride = byte distance along each axis
    strides = arr.strides + arr.strides

    # ? create different view on the same memory
    sub_matrices = np.lib.stride_tricks.as_strided(arr, view_shape, strides)
    return sub_matrices


# ----------------------------- Dilation/Erosion ----------------------------- #


def __get_orders(img, se: np.ndarray, order_cls: ColorOrder | None = None):
    if is_gray(img) or is_bool(img):
        return img
    assert is_color(img)
    if order_cls is None:
        order_cls = get_default_order()
    order_cls = BasicOrder()
    return order_cls.to_orders(img, se)


def __shift_and_combine(
    img: np.ndarray,
    se: np.ndarray,
    orders: np.ndarray,
    combine_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Internal function.
    """
    # write_ref(orders)

    assert len(orders.shape) == 2
    assert len(se.shape) == 2
    assert img.shape[:2] == orders.shape[:2]

    se = se.astype(bool)
    cx, cy = se.shape[0] // 2, se.shape[1] // 2

    x, y = np.nonzero(se)
    shift_x = x - cx
    shift_y = y - cy
    xs, ys = np.mgrid[: orders.shape[0], : orders.shape[1]]
    xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, orders.shape[0] - 1)
    ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, orders.shape[1] - 1)
    order_layers = orders[xs, ys]
    # imgs = imgs.swapaxes(-1, -2)
    assert order_layers.shape == orders.shape + x.shape, order_layers.shape

    # idxs: np.ndarray = np.argmin(imgs, axis=-1)
    idxs = combine_func(order_layers)
    xs, ys = np.mgrid[: orders.shape[0], : orders.shape[1]]
    xs: np.ndarray = np.clip(xs - shift_x[idxs], 0, orders.shape[0] - 1)
    ys: np.ndarray = np.clip(ys - shift_y[idxs], 0, orders.shape[1] - 1)
    res = img[xs, ys]
    return res



def __shift_and_combine_fuzzy(
    img: np.ndarray,
    se: np.ndarray,
    orders: np.ndarray,
    mode: Literal["dilation", "erosion"],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Internal function.
    """

    assert len(orders.shape) == 2
    assert len(se.shape) == 2
    assert img.shape[:2] == orders.shape[:2]

    se = se.astype(bool)
    cx, cy = se.shape[0] // 2, se.shape[1] // 2

    x, y = np.nonzero(se)
    shift_x = x - cx
    shift_y = y - cy
    xs, ys = np.mgrid[: orders.shape[0], : orders.shape[1]]
    xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, orders.shape[0] - 1)
    ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, orders.shape[1] - 1)
    order_layers = orders[xs, ys]
    assert order_layers.shape == orders.shape + x.shape, order_layers.shape

    # first normalize orders
    order_layers = np.argsort(order_layers, axis=-1)
    assert np.all(order_layers < order_layers.shape[-1])

    if mode == "erosion":
        order_layers = -order_layers

    origin = img[xs, ys]
    assert origin.shape[:3] == order_layers.shape
    weights = np.exp(alpha * order_layers)
    if len(weights.shape) < len(origin.shape):
        weights = weights[..., None]
    res: np.ndarray = np.sum(origin * weights, axis=2) / np.sum(weights, axis=2)
    assert res.shape == img.shape, res.shape

    return res


def dilation(
    img: np.ndarray,
    se: np.ndarray,
    order_cls: ColorOrder | None = None,
    fuzzy=False,
    alpha: float = 0.5,
) -> np.ndarray:
    if not fuzzy:
        return __shift_and_combine(
            img,
            se,
            __get_orders(img, se, order_cls),
            lambda x: np.argmax(x, axis=-1),
        )
    return __shift_and_combine_fuzzy(img, se, __get_orders(img, se, order_cls), "dilation", alpha)


def erosion(
    img: np.ndarray,
    se: np.ndarray,
    order_cls: ColorOrder | None = None,
    fuzzy=False,
    alpha: float = 0.5,
) -> np.ndarray:
    if not fuzzy:
        return __shift_and_combine(
            img,
            se,
            __get_orders(img, se, order_cls),
            lambda x: np.argmin(x, axis=-1),
        )

    return __shift_and_combine_fuzzy(img, se, __get_orders(img, se, order_cls), "dilation", alpha)


# ------------------------------ Opening/Closing ----------------------------- #


def opening(
    img: np.ndarray,
    se: np.ndarray,
    order_cls: ColorOrder | None = None,
    fuzzy=False,
    alpha: float = 0.5,
) -> np.ndarray:
    return dilation(
        erosion(img, se, order_cls, fuzzy, alpha),
        se,
        order_cls,
        fuzzy,
        alpha,
    )


def closing(
    img: np.ndarray,
    se: np.ndarray,
    order_cls: ColorOrder | None = None,
    fuzzy=False,
    alpha: float = 0.5,
) -> np.ndarray:
    return erosion(
        dilation(img, se, order_cls, fuzzy, alpha),
        se,
        order_cls,
        fuzzy,
        alpha,
    )
