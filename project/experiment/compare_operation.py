"""
Compare the effect on original image (without noise).
"""

from morphology import SE, _shift_and_combine_fuzzy, closing, dilation, erosion, opening
from order import HSVOrder, RGBOrder, RGBOrderMode
from report_utils import Figure, figure_end, figure_start, img2subfigure, write_report
from requests import get
from sample import Sample, get_sample
from task import task, write_ref
from tqdm import tqdm
from utils import rgb2gray


@task()
def compare_mode1():
    lena = get_sample(Sample.LENA)

    figure = Figure("Comparison of different modes")

    mode: RGBOrderMode
    for mode in ("sum", "prod", "median"):
        for fn in (dilation, erosion, opening, closing):
            mode_name = mode if mode != "prod" else "product"
            [name] = write_ref(fn(lena, SE.circle(2), RGBOrder(mode)), f"{fn.__name__}_{mode}")
            figure.add_subfigure(name, 0.22, f"{fn.__name__} with {mode_name}")

    figure.write("compare_mode1")


@task()
def compare_order():
    lena = get_sample(Sample.LENA)

    figure = Figure("Comparison of proposed ordering, HSV and gray scale")

    pbar = tqdm(total=12)
    order_zip = zip((RGBOrder("sum"), HSVOrder()), ("proposed", "HSV"))
    for order, order_name in order_zip:
        for fn in (dilation, erosion):
            filename = f"{fn.__name__}_{order_name}"
            caption = f"{fn.__name__} with {order_name}"
            [name] = write_ref(fn(lena, SE.circle(2), order), filename)
            figure.add_subfigure(name, 0.22, caption)
            pbar.update(1)

            filename = f"{fn.__name__}_{order_name}_fuzzy"
            caption = f"{fn.__name__} with {order_name} (fuzzy)"
            [name] = write_ref(fn(lena, SE.circle(2), order, fuzzy=True), filename)
            figure.add_subfigure(name, 0.22, caption)
            pbar.update(1)

    for fn in (dilation, erosion):
        order_name = "gray scale"

        filename = f"{fn.__name__}_{order_name}"
        caption = f"{fn.__name__} with {order_name}"
        [name] = write_ref(fn(rgb2gray(lena), SE.circle(2)), filename)
        figure.add_subfigure(name, 0.22, caption)
        pbar.update(1)

        filename = f"{fn.__name__}_{order_name}_fuzzy"
        caption = f"{fn.__name__} with {order_name} (fuzzy)"
        [name] = write_ref(fn(rgb2gray(lena), SE.circle(2), fuzzy=True), filename)
        figure.add_subfigure(name, 0.22, caption)
        pbar.update(1)

    figure.write("compare_order")


@task()
def compare_global():
    lena = get_sample(Sample.LENA)
    order_local = RGBOrder("sum")
    order_global = RGBOrder("sum", is_global=True)
    fn = dilation
    lena_local = fn(lena, SE.circle(2), order_local)
    lena_local_fuzzy = fn(lena, SE.circle(2), order_local, fuzzy=True)
    lena_global = fn(lena, SE.circle(2), order_global)
    lena_global_fuzzy = fn(lena, SE.circle(2), order_global, fuzzy=True)
    lena_local_argsort = order_local.dilation(lena, SE.circle(2), True)
    lena_local_argsort_fuzzy = _shift_and_combine_fuzzy(
        lena, SE.circle(2), order_local.to_order_layers(lena, SE.circle(2), True), 0.5
    )

    imgs = [
        lena_local,
        lena_local_fuzzy,
        lena_local_argsort,
        lena_local_argsort_fuzzy,
        lena_global,
        lena_global_fuzzy,
    ]
    paths = write_ref(imgs)
    captions = [
        "Local orders",
        "Local orders (fuzzy)",
        "Local orders (argsort)",
        "Local orders (argsort, fuzzy)",
        "Global orders",
        "Global orders (fuzzy)",
    ]

    figure = Figure("Comparison of dilation with global and local orders")
    for path, caption in zip(paths, captions):
        figure.add_subfigure(path, 0.45, caption)
    figure.write("compare_global")
