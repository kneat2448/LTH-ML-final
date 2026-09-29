import datetime as dt

from src.utils import union_hours


def test_colab_runtime_is_union_of_intervals():
    t = lambda h, m=0: dt.datetime(2026, 1, 1, h, m)
    # three parallel chains overlapping from 1:00 to 3:00, then one alone 4:00-4:30
    iv = sorted([(t(1), t(3)), (t(1), t(2, 30)), (t(1, 30), t(3)), (t(4), t(4, 30))])
    assert abs(union_hours(iv) - 2.5) < 1e-9
    assert union_hours([]) == 0.0
