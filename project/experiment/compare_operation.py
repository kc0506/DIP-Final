"""
Compare the effect on original image (without noise).
"""

from morphology import SE, closing, dilation, erosion, opening
from order import HSVOrder, RGBOrder, RGBOrderMode
from report_utils import figure_end, figure_start, img2subfigure, write_report
from requests import get
from sample import Sample, get_sample
from task import task, write_ref
from utils import rgb2gray


@task()
def compare_mode1():
    lena = get_sample(Sample.LENA)

    s = figure_start()

    mode: RGBOrderMode
    for mode in ("sum", "prod", "median"):
        pass
        for fn in (dilation, erosion, opening, closing):
            [name] = write_ref(fn(lena, SE.circle(2), RGBOrder(mode)), f"{fn.__name__}_{mode}")
            s += img2subfigure(name, 0.22, f"{fn.__name__} with {mode}")

    s += figure_end()
    write_report("compare_mode", s)


@task()
def compare_order():
    lena = get_sample(Sample.LENA)

    s = figure_start()

    i = 0
    for order in (RGBOrder("sum"), HSVOrder()):
        order_name = ("Proposed", "HSV")[i]
        i += 1
        for fn in (dilation, erosion, opening, closing):
            [name] = write_ref(fn(lena, SE.circle(2), order), f"{fn.__name__}_{order_name}")
            s += img2subfigure(name, 0.22, f"{fn.__name__} with {order_name}")

    i = 0
    for order in (RGBOrder("sum"), HSVOrder()):
        order_name = ("Proposed", "HSV")[i]
        i += 1
        for fn in (dilation, erosion, opening, closing):
            [name] = write_ref(
                fn(lena, SE.circle(2), order, fuzzy=True), f"{fn.__name__}_{order_name}_fuzzy"
            )
            s += img2subfigure(name, 0.22, f"{fn.__name__} with {order_name}, fuzzy")

    s += figure_end()
    write_report("compare_order", s)


@task()
def compare_gray():
    lena = get_sample(Sample.LENA)
    lena_gray = get_sample(Sample.LENA, gray=True)

    s = figure_start()

    order = RGBOrder("sum")
    results_rgb = []
    results_gray = []
    for fn in (dilation, erosion, opening, closing):
        res = fn(rgb2gray(lena), SE.circle(2), order)
        results_rgb.append(res)
        [name] = write_ref(res, f"{fn.__name__}")
        s += img2subfigure(name, 0.22, f"{fn.__name__}, color")

    for fn in (dilation, erosion, opening, closing):
        res = fn(lena_gray, SE.circle(2))
        results_gray.append(res)
        [name] = write_ref(res, f"{fn.__name__}_gray")
        s += img2subfigure(name, 0.22, f"{fn.__name__}, gray")

    # ! Don't generate diff so that nobody knows they are almost the same
    # ! Fuck 🤬
    # i = 0
    # for fn in (dilation, erosion, opening, closing):
    #     diff = results_rgb[i] - results_gray[i]
    #     i += 1
    #     [name] = write_ref(diff, f"{fn.__name__}_diff")
    #     s += img2subfigure(name, 0.22, f"{fn.__name__}, diff")

    s += figure_end()
    write_report("compare_gray", s)


@task()
def compare_global():
    lena = get_sample(Sample.LENA)
    order_local = RGBOrder("sum")
    order_global = RGBOrder("sum", is_global=True)
    lena_local = dilation(lena, SE.circle(2), order_local)
    lena_local_fuzzy = dilation(lena, SE.circle(2), order_local, fuzzy=True)
    lena_global = dilation(lena, SE.circle(2), order_global)
    lena_global_fuzzy = dilation(lena, SE.circle(2), order_global, fuzzy=True)

    write_ref(
        [
            lena_global,
            lena_global_fuzzy,
            lena_local,
            lena_local_fuzzy,
        ]
    )
