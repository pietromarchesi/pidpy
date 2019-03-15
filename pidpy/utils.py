import numpy as np
import math
import warnings

try:
    import pymorton
except ImportError:
    warnings.warn("Module pymorton is not available, functionality"
                  "is limited to binary data.")

from pidpy.utilsc import _map_binary_array_par
from pidpy.utilsc import _compute_joint_probability_bin
from pidpy.utilsc import _compute_joint_probability_nonbin
from pidpy.utilsc import _map_binary


def _map_array(X, binary = True):
    '''
    High-level mapping function used in PIDCalculator.
    '''
    if binary:
        Xmap = _map_binary_array_par(X)
    else:
        Xmap = _map_nonbinary_array(X)
    return Xmap


def _map_nonbinary_array(X):
    '''
    Array of integer non-binary values to vector of
    integers using Morton encoding of every row of the input
    array. Supports only up to three variables.
    '''
    # TODO: this is fine but every time the value error is raised pymorton prints
    # stuff, which we don't want.
    # Xmap = np.zeros(X.shape[0], dtype = int)
    # for i in range(X.shape[0]):
    #     if X.shape[1] == 3:
    #         try:
    #             Xmap[i] = pymorton.interleave(X[i,0], X[i,1], X[i,2])
    #         except ValueError:
    #             Xmap[i] = pymorton.interleave(int(X[i,0]), int(X[i,1]), int(X[i,2]))
    #     if X.shape[1] == 2:
    #         try:
    #             Xmap[i] = pymorton.interleave(X[i,0], X[i,1])
    #         except ValueError:
    #             Xmap[i] = pymorton.interleave(int(X[i, 0]), int(X[i, 1]))
    # return Xmap

    Xmap = np.zeros(X.shape[0], dtype = int)
    for i in range(X.shape[0]):
        if X.shape[1] == 3:
            Xmap[i] = pymorton.interleave(int(X[i,0]), int(X[i,1]), int(X[i,2]))
        if X.shape[1] == 2:
            Xmap[i] = pymorton.interleave(int(X[i, 0]), int(X[i, 1]))
    return Xmap


def lazy_property(fn):
    '''
    Decorator used to ensure that relevant probability tables
    and other attributes are not recomputed if they have already
    been calculated.
    '''
    attr_name = '_lazy_' + fn.__name__
    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            #if getattr(self, 'verbosity') > 0:
            #    print('Computing %s' %fn.__name__)
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def _group_without_unit(group, unit):
    '''
    Returns the tuple given by `group without the element give by `uni`.
    '''
    if isinstance(unit, int):
        unit = [unit]
    return tuple(k for k in group if not k in unit)


def _get_surrogate_val(surrogate, fun, **kwargs):
    return getattr(surrogate, fun)(**kwargs)


def _isbinary(X):
    return set(X.flatten()) == {0,1}


def _joint_probability(X, y, binary = True):
    if X.ndim > 1:
        Xmap = _map_array(X, binary = binary)
        N = X.shape[1]
    else:
        Xmap = X
        N = 1

    if binary and N < 12:
        nvals = 2 ** N
        joint = _compute_joint_probability_bin(Xmap,y,nvals)
    else:
        joint = _compute_joint_probability_nonbin(Xmap, y)

    return joint


def _joint_var(X, y, binary = True):
    joints = []
    for i in range(X.shape[1]):
        joints.append(_joint_probability(X[:, i], y, binary = binary))
    return joints


def _joint_sub(X, y, binary = True):
    joints = []
    for i in range(X.shape[1]):
        group = _group_without_unit(range(X.shape[1]), i)
        joints.append(_joint_probability(X[:, group], y, binary = binary))
    return joints


def _Imin(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i]*np.min(spec_info[i])
    return Im


def _Imax(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i]*np.max(spec_info[i])
    return Im


def _conditional_probability_from_joint(joint):
    X_mar = joint.sum(axis = 1)
    y_mar = joint.sum(axis = 0)

    cond_Xy = joint.astype(float) / y_mar[np.newaxis, :]
    cond_yX = joint.astype(float) / X_mar[:, np.newaxis]
    return cond_Xy, cond_yX


#-------------------------------------------------------------
# The functions below are not currently in use.

def map_binary(x):
    '''
    Map vector of binary digits to the corresponding integer in the binary system.
    '''
    return sum(1 << i for i, b in enumerate(np.flipud(x)) if b)


def _map_binary_array(X, binary = True):
    '''
    Maps binary array. Old version in which mapping of the single
    rows of `X` is performed by the Cython function _map_binary
    '''
    mapped = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        mapped[i] = _map_binary(X[i,:])
    return mapped


def _map_binary_array_original(X, binary = True):
    '''
    Old function, used only for profiling purposes.
    '''
    mapped = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        mapped[i] = map_binary(X[i,:])

    return mapped


def cantor_pairing_function(k1, k2, safe=True):
    '''
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    '''
    z = int(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2)
    if safe and (k1, k2) != depair(z):
        raise ValueError("{} and {} cannot be paired".format(k1, k2))
    return z


def depair(z):
    '''
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    '''
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w ** 2 + w) / 2
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
