import numpy as np


def sliding_windows(x, length, hop):
    """Split a 1-D signal into overlapping windows, one per row.

    Trailing samples that don't fill a whole window are dropped.
    """
    x = np.asarray(x)
    if length <= 0 or hop <= 0:
        raise ValueError("length and hop must be positive")
    if x.size < length:
        return np.empty((0, length), dtype=x.dtype)
    n = 1 + (x.size - length) // hop
    idx = np.arange(length)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]
