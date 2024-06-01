import os
from time import time

import cv2
import numpy as np
from experiment.compare_operation import (
    compare_global,
    compare_gray,
    compare_mode1,
    compare_order,
)
from experiment.denoise import denoise
from median import median_filter
from morphology import SE, dilation, erosion, opening

# from hypergraph import HG_dilation, HG_reosion, HG_opening, HG_closing
from noise import gaussian_noise, uniform_pepper, uniform_salt
from order import BasicOrder, HSVOrder, RGBOrder
from report_utils import figure_end, figure_start, img2subfigure, write_report
from sample import Sample, get_sample, sample_path
from task import get_func_name, task, write_ref
from utils import read_img, rgb2hsv, write_img

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
        lena =lena+ uniform_salt(lena.shape, 0.1)
        lena = read_img(os.path.join( sample_path() , 'impulse.png')).astype(np.uint8)
           
        write_ref(erosion(lena, SE.circle(2), BasicOrder()),)

    # lena = get_sample(Sample.LENA, gray=True)
    # write_ref(dilation(lena, SE.circle(3)))
    task1()


# example()
write_ref( RGBOrder('sum').dilation(get_sample(Sample.LENA), SE.circle(3)) , 'fuck')

# compare_mode1()
# compare_order()
# compare_gray()
# denoise()
# compare_global()

# lena = get_sample(Sample.LENA)
# lena_hsv = rgb2hsv(lena)
# write_ref(dilation(lena, SE.circle(3), HSVOrder(), fuzzy=True))
