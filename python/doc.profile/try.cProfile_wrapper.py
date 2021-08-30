import cProfile, pstats, io
from pstats import SortKey

import psutil
import pandas as pd
import numpy as np

pr = cProfile.Profile()

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper

@profile
def test(ns):
    np.sort(ns)

N = 20000000
ns = np.random.randint(0, high=N, size=N, dtype=int)

test(ns)

