import cv2
from morphology import SE, dilation, erosion
from sample import Sample, get_sample
from task import get_func_name, task, write_ref
from utils import write_img

# sample = get_sample(Sample.LENA, gray=True)
# write_img("lena_gray", erosion(sample, SE.circle(3)))


@task()
def example():

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

example()
