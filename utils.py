import numpy as np


def calc_cross_long_liq_price(balance, pos_size, pos_price, mm=0.004) -> float:
    d = (pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance - pos_size * pos_price) / d


def calc_cross_shrt_liq_price(balance, pos_size, pos_price, mm=0.004) -> float:
    abs_pos_size = abs(pos_size)
    d = (abs_pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance - pos_size * pos_price) / d


def round_dn(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.floor(n / step) * step, safety_rounding)


def calc_diff(x, y):
    return abs(x - y) / abs(y)
