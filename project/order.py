from abc import abstractmethod
from typing import Callable, Literal, override

import cv2
import numpy as np
from task import write_ref
from utils import is_color, rgb2hsv


class ColorOrder:
    @abstractmethod
    def to_orders(self, img: np.ndarray, se: np.ndarray) -> np.ndarray: ...


class BasicOrder(ColorOrder):
    @override
    def to_orders(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        return img.sum(axis=-1)


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
        mode: RGBOrderMode,
        custom_fn: Callable | None = None,
        is_global=False,
    ) -> None:
        self.mode = mode
        self.custom_fn = custom_fn
        self.is_global = is_global
        # print(is_global)

    @override
    def to_orders(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        assert is_color(img)

        order3 = np.argsort(img.reshape((img.shape[0] * img.shape[1], 3)), axis=0)
        order3 = order3.reshape(img.shape)
        assert order3.shape == img.shape

        if not self.is_global:
            # print('hi')
            # adjust order to local one
            # ! not sure this is stable yet
            se = se.astype(bool)
            cx, cy = se.shape[0] // 2, se.shape[1] // 2
            x, y = np.nonzero(se)
            center_idx = [k for  k in range(len(x)) if x[k]==cx and y[k]==cy][0]
            print(len(x))

            shift_x = x - cx
            shift_y = y - cy
            xs, ys = np.mgrid[: order3.shape[0], : order3.shape[1]]
            xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, order3.shape[0] - 1)
            ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, order3.shape[1] - 1)
            order3_expand = order3[xs, ys]
            for c in range(3):
                print(order3_expand[..., c].argsort(axis=-1)[0,0])
                order3[..., c] = order3_expand[..., c].argsort(axis=-1)[..., center_idx]

        if self.custom_fn is not None:
            return self.custom_fn(order3)

        mode_fns: dict[RGBOrderMode, Callable] = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
            "median": lambda x: np.median(x, axis=-1),
        }

        return mode_fns[self.mode](order3)

    
    def dilation(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        
        img = np.zeros((20,20,3))
        cv2.circle(img, (10, 10), 3, (255,0 , 255), -1)
        assert is_color(img)
        write_ref(img)
        print(se)

        se = se.astype(bool)
        cx, cy = se.shape[0] // 2, se.shape[1] // 2
        x, y = np.nonzero(se)
        assert len(x) == se.sum()


        center_idx = [k for  k in range(len(x)) if x[k]==cx and y[k]==cy][0]
        shift_x = x - cx
        shift_y = y - cy

        print(shift_x)
        print(shift_y)
        xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
        xs: np.ndarray = np.clip(xs[:, :, np.newaxis] - shift_x, 0, img.shape[0] - 1)
        ys: np.ndarray = np.clip(ys[:, :, np.newaxis] - shift_y, 0, img.shape[1] - 1)
        return img[xs,ys].sum(axis=-2)/se.sum()
        # print(xs[0,0])
        # print(ys[0,0])

        img_layers = img[xs, ys]
        print(img_layers[10,5].sum(axis=-1))
        order_layers = np.argsort(img_layers, axis=2)
        print(order_layers[10,5])
        order_layers = np.sum(order_layers, axis=-1)

        idxs = np.argmax(order_layers, axis=-1)
        # print(idxs)

        xs, ys = np.mgrid[: img.shape[0], : img.shape[1]]
        xs: np.ndarray = np.clip(xs - shift_x[idxs], 0, img.shape[0] - 1)
        ys: np.ndarray = np.clip(ys - shift_y[idxs], 0, img.shape[1] - 1)
        res = img[xs, ys]
        return res

        order3 = np.argsort(img.reshape((img.shape[0] * img.shape[1], 3)), axis=0)
        order3 = order3.reshape(img.shape)
        assert order3.shape == img.shape

        # adjust order to local one
        # ! not sure this is stable yet
        print(len(x))

        order3_expand = order3[xs, ys]
        for c in range(3):
            print(order3_expand[..., c].argsort(axis=-1)[0,0])
            order3[..., c] = order3_expand[..., c].argsort(axis=-1)[..., center_idx]

        if self.custom_fn is not None:
            return self.custom_fn(order3)

        mode_fns: dict[RGBOrderMode, Callable] = {
            "sum": lambda x: np.sum(x, axis=-1),
            "prod": lambda x: np.prod(x, axis=-1),
            "harmonic": lambda x: 3 / np.sum(1 / x, axis=-1),
            "median": lambda x: np.median(x, axis=-1),
        }

        return mode_fns[self.mode](order3)



class HSVOrder(ColorOrder):
    @override
    def to_orders(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        img_hsv = rgb2hsv(img)
        max_h, max_s, max_v = (179, 255, 255)
        # lexigraphical order: V, S, H
        return np.sum(img_hsv * (max_h + max_s, max_h, 1), axis=-1)


def get_default_order() -> ColorOrder:
    # return (RGBOrder('sum'))
    return BasicOrder()
