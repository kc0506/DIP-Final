import numpy as np
from morphology import get_submat


def median_filter(
    img: np.ndarray,
    orders: np.ndarray,
    a: float,
    r: int,
    p=0.5,
) -> np.ndarray:
    """
    :param img: input image
    :param a: threshold
    :param r: window size
    """
    assert img.shape[:2] == orders.shape

    assert r % 2 == 1
    submat = get_submat(orders, (r, r))

    submat = submat.reshape((*submat.shape[:-2], -1))
    submat = submat.argsort(axis=-1)

    median_xs, median_ys = np.mgrid[: orders.shape[0], : orders.shape[1]]
    median_xs = get_submat(median_xs, (r, r)).reshape((*orders.shape, -1))
    median_ys = get_submat(median_ys, (r, r)).reshape((*orders.shape, -1))

    is_median = submat == r**2 / 2
    median_xs &= is_median
    median_xs = np.max(median_xs, axis=-1)
    median_ys &= is_median
    median_ys = np.max(median_ys, axis=-1)
    median_orders = orders[median_xs, median_ys]

    assert np.all(is_median >= 0)
    assert np.all(orders >= 0)

    cond = np.logical_or(
        orders > a* median_orders,
        median_orders > a * orders,
    )
    res = np.where(cond[:, :, np.newaxis], img[median_xs, median_ys], img)
    return res
