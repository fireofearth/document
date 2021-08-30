import cProfile, pstats, io
from pstats import SortKey

import psutil
import pandas as pd
import numpy as np

pr = cProfile.Profile()

N = 20000000
ns = np.random.randint(0, high=N, size=N, dtype=int)

pr.enable()

np.sort(ns)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
