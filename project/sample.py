import enum
import os
from functools import cache
from typing import Literal

from numpy import ndarray
from utils import read_img, rgb2gray


class Sample(enum.Enum):
    LENA = "lena"


_PATH_DICT: dict[Sample, str] = {
    Sample.LENA: "lena.png",
}


def sample_path():
    return os.path.join(__file__, "../", "./images/samples")


def get_sample(name: Sample, gray=False) -> ndarray:
    img = read_img(os.path.join(__file__, "../", "./images/samples", _PATH_DICT[name]))
    if gray:
        return rgb2gray(img)
    return img


get_sample = cache(get_sample)
