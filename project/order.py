from abc import abstractmethod
from typing import override

from numpy import ndarray
from utils import is_color


class ColorOrder:
    @classmethod
    @abstractmethod
    def to_orders(cls, img: ndarray) -> ndarray: ...


class BasicOrder(ColorOrder):
    @classmethod
    @override
    def to_orders(cls, img: ndarray) -> ndarray:
        return img.sum(axis=-1)


def get_default_order() -> ColorOrder:
    return BasicOrder()
