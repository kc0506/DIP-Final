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


def figure_end(caption: str = ""):
    return (
        r""" \caption{%s}
 \end{figure}
"""
        % caption
    )


def img2subfigure(img_path: os.PathLike | str, size: float, caption: str | None = None):

    img_path = os.path.relpath(img_path, Path(REPORT_PATH).parent)
    img_path = img_path.replace("\\", "/")
    img_path = img_path.replace("_", "\\_")

    if caption is None:
        caption = os.path.basename(img_path).removesuffix(".png")
    caption = caption.replace("_", "\\_")

    return r"""\begin{subfigure}[t]{%.2f\textwidth}
    \includegraphics[width=0.9\linewidth]{%s}
    \caption{%s}
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
