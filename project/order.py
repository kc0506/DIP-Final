from abc import abstractmethod
from typing import Callable, Literal, override

import numpy as np
from utils import is_color


class ColorOrder:
    @abstractmethod
    def to_orders(self, img: np.ndarray) -> np.ndarray: ...


class BasicOrder(ColorOrder):
    @override
    def to_orders(self, img: np.ndarray) -> np.ndarray:
        return img.sum(axis=-1)


RGBOrderMode = Literal[
    "sum",
    "prod",
    "harmonic",
]


class RGBOrder(ColorOrder):

    mode: RGBOrderMode
    custom_fn: Callable | None

    def __init__(self, mode: RGBOrderMode, custom_fn: Callable | None = None) -> None:
        self.mode = mode
        self.custom_fn = custom_fn

    @override
    def to_orders(self, img: np.ndarray) -> np.ndarray:
        order3 = np.argsort(img.reshape((img.shape[0] * img.shape[1], 3)), axis=0)
        order3 = order3.reshape(img.shape)
        assert order3.shape == img.shape

        if self.custom_fn is not None:
            return self.custom_fn(order3)

        mode_fns = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
        }
        return mode_fns[self.mode](order3)


def get_default_order() -> ColorOrder:
    return BasicOrder()
