import ast
import inspect
import os
import pathlib
import types
from functools import wraps
from os import PathLike
from typing import Callable, Sequence

import executing
import numpy as np
from utils import write_img


def get_func_name(depth=0):
    return inspect.stack()[depth + 1].function


_default_cnt = 0


def get_arg_names(frame: types.FrameType, pos=0) -> list[str]:
    try:
        node = executing.Source.executing(frame).node
        arg = node.args[pos]  # type: ignore

        def get_id(a) -> list[str]:
            if isinstance(a, ast.List):
                return [get_id(_a)[0] for _a in a.elts]
            if isinstance(a, ast.Name):
                return [a.id]
            if isinstance(a, ast.Call):
                if len(a.args) == 0:
                    return [a.func.id]  # type: ignore
                return [get_id(a.args[0])[0] + "_" + a.func.id]  # type: ignore
            if isinstance(a, ast.ListComp):
                raise Exception()
            if isinstance(a, ast.Attribute):
                return [get_id(a.value)[0]]
            if isinstance(a, ast.BinOp):
                return [get_id(a.left)[0]]
            if isinstance(a, ast.Subscript):
                return [get_id(a.value)[0]]
            print("warning:", type(a))
            raise Exception()

        return get_id(arg)

    except:
        global _default_cnt
        _default_cnt += 1
        return [f"default{_default_cnt}"]


__OUT_ATTR = "__ref__"


def get_out_dir() -> pathlib.Path:
    path = pathlib.Path(__file__).parent / "./images/outputs"
    path_rev = pathlib.Path()
    for s in inspect.stack()[1:]:
        frame = s.frame
        func = frame.f_globals.get(s.function, None) or frame.f_locals.get(s.function, None)
        if not hasattr(func, __OUT_ATTR):
            continue

        path_rev /= getattr(func, __OUT_ATTR)
    # convert path_rev to list
    for p in reversed(path_rev.parts):
        path /= p
    return path


def write_ref(img: np.ndarray | list[np.ndarray], _name: str | list[str] = ""):
    """
    Allow formats:
    - write_ref(img)
    - write_ref([img1, img2])
    - write_ref(imgs)
    - write_ref(f(img))
    - write_ref(img, "name")
    - write_ref([img1, img2], ["name1", "name2"])
    """

    frame: types.FrameType = inspect.currentframe().f_back  # type: ignore

    if _name:
        if isinstance(_name, str):
            _name = [_name]
        names = _name
    else:
        names = get_arg_names(frame)
    assert len(names)

    out_dir = get_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(img, np.ndarray):
        write_img(out_dir / names[0], img)
        return [out_dir / (n + ".png") for n in names]

    if len(names) != len(img):
        names = [f"{names[0]}_{i}" for i in range(len(img))]
    assert len(names) == len(img)
    for name, i in zip(names, img):
        write_img(out_dir / name, i)

    return [out_dir / (n + ".png") for n in names]


def task(parent=None, dir_name=None):
    def dec[T, **P](f: Callable[P, T]):

        # @wraps(f)
        def _f(*args: P.args, **kwargs: P.kwargs) -> T:

            # print("🚩", f.__name__, "started")
            res = f(*args, **kwargs)
            # print("✅", f.__name__, "completed")
            # print()

            return res

        _f = wraps(f)(_f)

        path = dir_name or f.__name__
        setattr(_f, __OUT_ATTR, path)
        return _f

    return dec
