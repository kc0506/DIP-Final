from morphology import SE, closing, dilation, erosion, opening
from noise import uniform_pepper, uniform_salt
from order import RGBOrder
from sample import Sample, get_sample
from task import task, write_ref


@task()
def erosion_exp():
    lena = get_sample(Sample.LENA)
    lenas = [erosion(lena, SE.circle(i), RGBOrder("prod")) for i in (1, 3, 5)]
    names = [f"lena_erosion{i}" for i in (1, 3, 5)]
    write_ref(lenas, names)


@task()
def dilation_exp():
    lena = get_sample(Sample.LENA)
    lenas = [dilation(lena, SE.circle(i), RGBOrder("prod")) for i in (1, 3, 5)]
    names = [f"lena_erosion{i}" for i in (1, 3, 5)]
    write_ref(lenas, names)


@task()
def noise_exp():
    lena = get_sample(Sample.LENA)
    lena_salt = uniform_salt(lena.shape) + lena
    lena_pepper = uniform_pepper(lena.shape) + lena
    write_ref(
        [
            lena_salt,
            opening(lena_salt, SE.circle(1), RGBOrder("prod")),
            closing(lena_salt, SE.circle(1), RGBOrder("prod")),
            lena_pepper,
            opening(lena_pepper, SE.circle(1), RGBOrder("prod")),
            closing(lena_pepper, SE.circle(1), RGBOrder("prod")),
        ]
    )


def main():
    erosion_exp()
    dilation_exp()

    noise_exp()


main()
