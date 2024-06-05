import enum
from ctypes.wintypes import RGB
from os import write
from pathlib import Path
from time import time

from hypergragh import HG_closing, HG_opening
from matplotlib import figure
from morphology import SE, closing, opening
from order import BasicOrder, RGBOrder
from report_utils import (
    Figure,
    Subfigure,
    figure_end,
    figure_start,
    img2subfigure,
    write_report,
)
from sample import get_noises, sample_path
from task import task, write_ref
from tqdm import tqdm
from utils import read_img


@task()
def denoise():
    # hg()

    # co()
    # oc()
    # before()
    pass

@task()
def before():
    sub = Subfigure("Noisy images")
    for i, noise in enumerate(tqdm(get_noises())):
        [name] = write_ref(noise, f"before{i}")
        sub.add_subfigure(name, 0.15)
    sub.write("before_noise")
        


@task()
def co():
    t = time()
    fig = Subfigure("close-opening with proposed method")
    order = RGBOrder("sum")
    se = SE.circle(2)
    for i, noise in enumerate(tqdm(get_noises())):
        deno = opening(closing(noise, se, order), se, order)
        [name] = write_ref(deno, f"co{i}")
        fig.add_subfigure(name, 0.15)
    fig.write("co_noise")
    print(time() - t, "complete CO")


@task()
def oc():
    t = time()
    fig = Subfigure("open-closing with proposed method")
    order = RGBOrder("sum")
    se = SE.circle(2)
    for i, noise in enumerate(tqdm(get_noises())):
        deno = closing(opening(noise, se, order), se, order)
        [name] = write_ref(deno, f"oc{i}")
        fig.add_subfigure(name, 0.15)
    fig.write("oc_noise")
    print(time() - t, "complete oc")


@task()
def hg():
    t = time()
    fig = Subfigure("Wang's close-opening")
    for i, noise in enumerate(tqdm(get_noises())):
        deno = HG_opening(HG_closing(noise))
        [name] = write_ref(deno, f"hg{i}")
        fig.add_subfigure(name, 0.15)
    fig.write("hg_noise")
    print(time() - t, "complete HG")
