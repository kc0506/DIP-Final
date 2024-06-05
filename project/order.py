from abc import abstractmethod
from operator import is_
from typing import Callable, Literal, override

import cv2
import numpy as np
from task import write_ref
from utils import get_layers, is_color, remap, rgb2hsv


class ColorOrder:
    @abstractmethod
    def to_order_layers(self, img: np.ndarray, se: np.ndarray) -> np.ndarray: ...


class BasicOrder(ColorOrder):
    @override
    def to_order_layers(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        return remap(get_layers(img.sum(axis=-1), se), 0, se.astype(bool).sum())


RGBOrderMode = Literal[
    "sum",
    "prod",
    "harmonic",
    "median",
]


class RGBOrder(ColorOrder):

    mode: RGBOrderMode
    custom_fn: Callable | None

    def __init__(
        self,
        mode: RGBOrderMode = "sum",
        custom_fn: Callable | None = None,
        is_global=False,
    ) -> None:
        self.mode = mode
        self.custom_fn = custom_fn
        self.is_global = is_global
        # print(is_global)

    # @override
    def get_global_orders(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        assert is_color(img)

        order3 = np.argsort(img.reshape((img.shape[0] * img.shape[1], 3)), axis=0)
        order3 = order3.reshape(img.shape)
        assert order3.shape == img.shape

        if self.custom_fn is not None:
            return self.custom_fn(order3)

        mode_fns: dict[RGBOrderMode, Callable] = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
            "median": lambda x: np.median(x, axis=-1),
        }

        orders = mode_fns[self.mode](order3)
        layers = get_layers(orders, se)
        return (layers - layers.min()) / (layers.max() - layers.min()) * se.astype(bool).sum()

    @override
    def to_order_layers(self, img: np.ndarray, se: np.ndarray, use_argsort=False) -> np.ndarray:
        if self.is_global:
            return self.get_global_orders(img, se)

        se = se.astype(bool)
        cx, cy = se.shape[0] // 2, se.shape[1] // 2
        x, y = np.nonzero(se)
        assert len(x) == se.sum()

        shift_x = x - cx
        shift_y = y - cy

        xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
        xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, img.shape[0] - 1)
        ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, img.shape[1] - 1)

        img_layers = img[xs, ys]

        order_layers = np.zeros_like(img_layers)
        for i in range(se.sum()):
            # cur = img_layers[:, :, i]
            # x = cur[:, :, np.newaxis] > img_layers
            # order_layers[:, :, i] = x.sum(axis=2)
            for c in range(3):
                cur = img_layers[:, :, i, c]
                x = cur[:, :, np.newaxis] > img_layers[..., c]
                order_layers[:, :, i, c] = x.sum(axis=2)
        if use_argsort:
            order_layers = np.argsort(img_layers, axis=2)

        mode_fns: dict[RGBOrderMode, Callable] = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
            "median": lambda x: np.median(x, axis=-1),
        }
        order_reduced: np.ndarray = mode_fns[self.mode](order_layers)
        return order_reduced

    def _apply(
        self, img: np.ndarray, se: np.ndarray, is_dilation: bool, use_argsort=False
    ) -> np.ndarray:
        se = se.astype(bool)
        cx, cy = se.shape[0] // 2, se.shape[1] // 2
        x, y = np.nonzero(se)
        assert len(x) == se.sum()

        shift_x = x - cx
        shift_y = y - cy

        xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
        xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, img.shape[0] - 1)
        ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, img.shape[1] - 1)

        img_layers = img[xs, ys]

        order_layers = np.zeros_like(img_layers)
        for i in range(se.sum()):
            cur = img_layers[:, :, i]
            if is_dilation:
                x = cur[:, :, np.newaxis] > img_layers
            else:
                x = cur[:, :, np.newaxis] < img_layers
            order_layers[:, :, i] = x.sum(axis=2)

        if use_argsort:
            order_layers = np.argsort(img_layers, axis=2)

        mode_fns: dict[RGBOrderMode, Callable] = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
            "median": lambda x: np.median(x, axis=-1),
        }
        order_reduced: np.ndarray = mode_fns[self.mode](order_layers)
        idxs: np.ndarray = np.argmax(order_reduced, axis=-1)

        xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
        xs: np.ndarray = np.clip(xs - shift_x[idxs], 0, img.shape[0] - 1)
        ys: np.ndarray = np.clip(ys - shift_y[idxs], 0, img.shape[1] - 1)
        res = img[xs, ys]
        return res

    def dilation(self, img: np.ndarray, se: np.ndarray, use_argsort=False) -> np.ndarray:
        return self._apply(img, se, True, use_argsort)

    def erosion(self, img: np.ndarray, se: np.ndarray, use_argsort=False) -> np.ndarray:
        return self._apply(img, se, False, use_argsort)


class HSVOrder(ColorOrder):
    @override
    def to_order_layers(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        img_hsv = rgb2hsv(img)
        max_h, max_s, max_v = (179, 255, 255)

        # # lexigraphical order: V, S, H
        lex = -np.sum(img_hsv * (max_h + max_s, max_h, 1), axis=-1)
        layers = get_layers(lex, se)
        layers = (layers - layers.min()) / (layers.max() - layers.min()) * se.astype(bool).sum()
        return layers


def get_default_order() -> ColorOrder:
    # return (RGBOrder('sum'))
    return BasicOrder()
