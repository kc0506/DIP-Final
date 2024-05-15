import enum
import os
from functools import cache
from typing import Literal

from numpy import ndarray
from utils import read_img, rgb_to_gray


class Sample(enum.Enum):
    LENA = "lena"


_PATH_DICT: dict[Sample, str] = {
    Sample.LENA: "lena.png",
}


def get_sample(name: Sample, gray=False) -> ndarray:
    img = read_img(os.path.join(__file__, "../", "./images/samples", _PATH_DICT[name]))
    if gray:
        return rgb_to_gray(img)
    return img


get_sample = cache(get_sample)
