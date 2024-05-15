from typing import Literal

import numpy as np


def gaussian_noise(
    shape: tuple[int, ...],
    mean: float = 0,
    std: float = 20,
) -> np.ndarray:
    return np.random.normal(mean, std, shape)


def uniform_pepper(
    shape: tuple[int, ...],
    p=0.01,
) -> np.ndarray:
    return np.where(np.random.sample(shape) <= p, -255, 0)


def uniform_salt(
    shape: tuple[int, ...],
    p=0.01,
) -> np.ndarray:
    return np.where(np.random.sample(shape) <= p, 255, 0)
