import numpy as np

from pidpy.utils import map_binary
from pidpy.utils import map_binary_array
from pidpy.utils import map_binary_array_original
from pidpy.utilsc import _map_binary, _map_binary1, _map_binary2
from pidpy.utilsc import _map_binary_array
from pidpy.utilsc import _map_binary_array_par


if False:
    x = np.array([1,0,1,0,0,0,0,0,1,0,1])
    mp = _map_binary(x)
    mc = _map_binary2(x)
    assert(mp == mc)
    %timeit map_binary(x)
    %timeit _map_binary(x)
    %timeit _map_binary1(x)
    %timeit _map_binary2(x)

import time



X = np.random.randint(2,size = [1000,11])
mp = map_binary_array(X)
mc = _map_binary_array(X)
mpp = _map_binary_array_par(X)

np.testing.assert_array_equal(mp,mpp)

%timeit map_binary_array_original(X)
%timeit map_binary_array(X)
%timeit _map_binary_array(X)
%timeit _map_binary_array_par(X)

