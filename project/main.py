from time import time

import cv2
import numpy as np
from median import median_filter
from morphology import SE, dilation, erosion
from noise import gaussian_noise, uniform_pepper, uniform_salt
from order import BasicOrder, RGBOrder
from sample import Sample, get_sample
from task import get_func_name, task, write_ref
from utils import write_img

# sample = get_sample(Sample.LENA, gray=True)
# write_img("lena_gray", erosion(sample, SE.circle(3)))


@task()
def example():
    """
    An example showing how to use utility functions.
    """

    @task()
    def task1():
        lena = get_sample(Sample.LENA)
        lena_erosion = [
            erosion(lena, SE.circle(3)),
            erosion(lena, SE.circle(5)),
        ]
        write_ref(lena_erosion)

    lena = get_sample(Sample.LENA, gray=True)
    write_ref(dilation(lena, SE.circle(3)))
    task1()
