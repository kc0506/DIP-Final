from time import time

from morphology import SE, closing, dilation, erosion, opening
from noise import uniform_pepper, uniform_salt
from order import RGBOrder
from report_utils import Figure, Subfigure
from sample import Sample, get_sample
from task import task, write_ref


@task()
def erosion_exp():
    t = time()
    lena = get_sample(Sample.LENA)
    lenas = [erosion(lena, SE.circle(i), RGBOrder("prod")) for i in (1, 3, 5)]
    names = [f"lena_erosion{i}" for i in (1, 3, 5)]
    paths = write_ref(lenas, names)

    sub = Subfigure("erosion with different SE")
    for path in paths:
        sub.add_subfigure(path, 0.2)
    sub.write("erosion")
    print(time() - t, "erosion")


@task()
def dilation_exp():
    t = time()
    lena = get_sample(Sample.LENA)
    lenas = [dilation(lena, SE.circle(i), RGBOrder("prod")) for i in (1, 3, 5)]
    names = [f"lena_dilation{i}" for i in (1, 3, 5)]
    paths = write_ref(lenas, names)

    sub = Subfigure("dilation with different SE")
    for path in paths:
        sub.add_subfigure(path, 0.2)
    sub.write("dilation")
    print(time() - t, "dilation")


@task()
def noise_exp():
    lena = get_sample(Sample.LENA)
    lena_salt = uniform_salt(lena.shape, 0.1) + lena
    lena_pepper = uniform_pepper(lena.shape, 0.1) + lena
    order = RGBOrder("sum")
    se = SE.circle(2)
    paths = write_ref(
        [
            lena_salt,
            opening(lena_salt, se, order),
            closing(lena_salt, se, order),
            lena_pepper,
            opening(lena_pepper, se, order),
            closing(lena_pepper, se, order),
        ]
    )
    captions = [
        "salt noise",
        "salt noise opening",
        "salt noise closing",
        "pepper noise",
        "pepper noise opening",
        "pepper noise closing",
    ]

    sub = Figure("pepper/salt noise")
    for path, caption in zip(paths, captions):
        sub.add_subfigure(path, 0.25, caption)
    sub.write("noise")


def main():
    erosion_exp()
    dilation_exp()

    noise_exp()


lena = get_sample(Sample.LENA)
lena_gray = erosion(lena, SE.circle(1), RGBOrder("median"))
write_ref(lena_gray)
write_ref(lena - lena_gray)

diff = lena - lena_gray
diff = opening(diff, SE.circle(2), RGBOrder("prod"))
write_ref(diff)
