import numpy as np
import pytest

from fencedet.windows import sliding_windows


def test_overlap_and_drop_tail():
    w = sliding_windows(np.arange(10), length=4, hop=2)
    assert w.shape == (4, 4)
    assert w[1].tolist() == [2, 3, 4, 5]
    assert w[-1].tolist() == [6, 7, 8, 9]


def test_short_signal_gives_no_windows():
    assert sliding_windows(np.arange(3), length=4, hop=1).shape == (0, 4)


def test_bad_hop():
    with pytest.raises(ValueError):
        sliding_windows(np.arange(10), length=4, hop=0)
