import inspect
import os
import types
from pathlib import Path

from numpy import ndarray
from task import get_arg_names, get_out_dir

REPORT_PATH = os.path.join(__file__, "../../report", "report.tex")


def figure_start():
    return r"""\begin{figure}[!ht]
   \centering
"""


def subfigure_start():
    return r"""\begin{subfigure}{0.9\textwidth}
   \centering
"""


def figure_end(caption: str = ""):
    return (
        r""" \caption{%s}
 \end{figure}
"""
        % caption
    )


def subfigure_end(caption: str = ""):
    return (
        r""" \caption{%s}
 \end{subfigure}
"""
        % caption
    )


def img2subfigure(img_path: os.PathLike | str, size: float, caption: str | None = None):

    img_path = os.path.relpath(img_path, Path(REPORT_PATH).parent)
    img_path = img_path.replace("\\", "/")
    img_path = img_path.replace("_", "\\_")

    # if caption is None:
    #     caption = os.path.basename(img_path).removesuffix(".png")
    if caption is not None:
        caption = caption.replace("_", "\\_")
        caption = r"\caption{%s}" % caption
    else:
        caption = ""

    return r"""\begin{subfigure}[t]{%.2f\textwidth}
    \includegraphics[width=0.9\linewidth]{%s}
    %s
    \centering
  \end{subfigure}
""" % (
        size,
        img_path,
        caption,
    )


def write_report(filename: str, content: str):
    if not filename.endswith(".tex"):
        filename += ".tex"
    with open(Path(REPORT_PATH).parent / filename, "w+") as f:
        f.write(content)
        f.close()


class Figure:
    def __init__(self, caption: str) -> types.NoneType:
        self.caption = caption
        self.__content = figure_start()

    def add_subfigure(self, img_path: os.PathLike | str, size: float, caption: str | None = None):
        self.__content += img2subfigure(img_path, size, caption)

    def write(self, filename: str):
        self.__content += figure_end(self.caption)
        write_report(filename, self.__content)
        self.__content = figure_start()


class Subfigure:
    def __init__(self, caption: str) -> types.NoneType:
        self.caption = caption
        self.__content = subfigure_start()

    def add_subfigure(self, img_path: os.PathLike | str, size: float, caption: str | None = None):
        self.__content += img2subfigure(img_path, size, caption)

    def write(self, filename: str):
        self.__content += subfigure_end(self.caption)
        write_report(filename, self.__content)
        self.__content = subfigure_start()
