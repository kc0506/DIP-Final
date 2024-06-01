from os import write
from pathlib import Path

from matplotlib import figure
from morphology import SE, closing, opening
from order import BasicOrder
from report_utils import figure_end, figure_start, img2subfigure, write_report
from sample import sample_path
from task import task, write_ref
from utils import read_img


@task()
def denoise():
    samples = Path(sample_path())
    samples /= "noise"

    s = figure_start()
    for img_path in samples.iterdir():
        img = read_img(str(img_path))
        name = img_path.stem
        [path] = write_ref(img, name)
        s += img2subfigure(path, 0.15, '')
    s += figure_end("Noised images")

    s += figure_start()
    for img_path in samples.iterdir():
        img = read_img(str(img_path))
        name = img_path.stem
        img = (opening(img, SE.circle(3), BasicOrder()) )
        [path] = write_ref(img, name + "_denoised")
        s += img2subfigure(path, 0.15, '')
    s += figure_end("Noised images with opening-closing")

    s += figure_start()
    for img_path in samples.iterdir():
        img = read_img(str(img_path))
        name = img_path.stem
        img = opening(img, SE.circle(1), fuzzy=True)  
        [path] = write_ref(img, name + "_denoised_fuzzy")
        s += img2subfigure(path, 0.15, '')
    s += figure_end("Noised images with fuzzy opening-closing")

    write_report("denoise", s)
