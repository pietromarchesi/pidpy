import numpy as np
import math
import pymorton
from pidpy.utilsc import _map_binary
from pidpy.utilsc import _map_binary_array
from pidpy.utilsc import _map_binary_array_par



def map_array(X, binary = True):
    '''
    High-level mapping function used in PIDCollector
    '''
    if binary:
        Xmap = _map_binary_array_par(X)
    else:
        Xmap = map_nonbinary_array(X)
    return Xmap


def map_nonbinary_array(X):
    '''
    Array of integer non-binary values to vector of
    integers using Morton encoding of every row of the input
    array.
    '''
    Xmap = np.zeros(X.shape[0], dtype = int)
    for i in range(X.shape[0]):
        if X.shape[1] == 3:
            Xmap[i] = pymorton.interleave(X[i,0], X[i,1], X[i,2])
        if X.shape[1] == 2:
            Xmap[i] = pymorton.interleave(X[i,0], X[i,1])

    return Xmap


from pidpy.utilsc import _map_binary
def map_binary_array(X, binary = True):
    '''
    Maps
    '''

    mapped = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        mapped[i] = _map_binary(X[i,:])

    return mapped

def map_binary_array_original(X, binary = True):
    # maps X using

    mapped = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        mapped[i] = map_binary(X[i,:])

    return mapped

def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            if getattr(self, 'verbosity') > 0:
                print('Computing %s' %fn.__name__)
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop

def group_without_unit(group, unit):
    """ Returns the tuple given by group without the element give by unit. """
    if isinstance(unit, int):
        unit = [unit]
    return tuple(k for k in group if not k in unit)

#-------------------------------------------------------------


def map_binary(x):
    # Map vector of binary digits to the integer in binary numeral system.
    return sum(1<<i for i, b in enumerate(np.flipud(x)) if b)

def cantor_pairing_function(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = int(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2)
    if safe and (k1, k2) != depair(z):
        raise ValueError("{} and {} cannot be paired".format(k1, k2))
    return z

def depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    # assert z != pair(x, y, safe=False):
    return x, y

def map_nonbinary(x):
    x = np.array(x)
    for i in range(len(x) - 1):
        # tup[i+1] = tup[i] * tup[i+1]
        x[i + 1] = cantor_pairing_function(x[i], x[i + 1])
    return x[i + 1]

def feature_values(X):
    # figure out whether X is binary or not, returning the
    # discrete feature values.
    nvals = len(set(X.reshape([X.shape[0] * X.shape[1], ])))

