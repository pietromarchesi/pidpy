import numpy as np

from pidpy.utils import map_binary
from pidpy.utilsc import _map_binary

x = np.array([1,0,1,0,1,1,0,1])
mp = map_binary(x)
mc = _map_binary(x)
assert(mp == mc)
%timeit map_binary(x)
%timeit _map_binary(x)

