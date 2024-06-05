import enum
import os
from functools import cache
from pathlib import Path
from typing import Literal

from numpy import ndarray
from utils import read_img, rgb2gray


class Sample(enum.Enum):
    LENA = "lena"
    IMPULSE = "impulse"


_PATH_DICT: dict[Sample, str] = {
    Sample.LENA: "lena.png",
    Sample.IMPULSE: "impulse.png",
}


def sample_path():
    return os.path.join(__file__, "../", "../images/samples")


def get_sample(name: Sample, gray=False) -> ndarray:
    img = read_img(os.path.join(__file__, "../", "../images/samples", _PATH_DICT[name]))
    if gray:
        return rgb2gray(img)
    return img


def get_noises():
    samples = Path(sample_path())
    samples /= "noise"
    imgs: list[ndarray] = []
    for img_path in samples.iterdir():
        img = read_img(str(img_path))
        imgs.append(img)
    return imgs[:6]

get_noises = cache(get_noises)
get_sample = cache(get_sample)
