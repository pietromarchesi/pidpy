#cython: profile=True
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport log2, fabs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_joint_probability_nonbin(np.ndarray[np.int64_t, ndim=1] X,
                               np.ndarray[np.int64_t, ndim=1] y):
    '''
    The joint probability table is built only with values which actually
    appear in X.
    '''
    cdef int nsamp  = y.shape[0]
    cdef int nlabels = len(list(set(y)))
    cdef np.ndarray[np.int64_t, ndim=1] vals
    vals   = np.array(sorted(list(set(X))))
    nvals  = vals.shape[0]
    cdef np.ndarray[np.int64_t, ndim=2] joint
    joint = np.zeros([nvals, nlabels], dtype = np.int64)

    for i in xrange(nsamp):
        for j in range(nvals):
            if X[i] == vals[j]:
                ind = j
        joint[ind, y[i]] += 1

    return joint / np.float(nsamp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_joint_probability_bin(np.ndarray[np.int64_t, ndim=1] X,
                                   np.ndarray[np.int64_t, ndim=1] y,
                                   int nvals):
    '''
    The joint probability table is set up with all possible values
    of X (faster, but requires construction of arrays which are too big
    if X is not binary).
    '''
    cdef int nsamp  = y.shape[0]
    cdef int nlabels = len(list(set(y)))
    cdef np.ndarray[np.int64_t, ndim=2] joint
    joint = np.zeros([nvals, nlabels], dtype = np.int64)

    for i in xrange(nsamp):
        joint[X[i], y[i]] += 1

    return joint / np.float(nsamp)



@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_mutual_info(np.ndarray[np.float64_t, ndim=1] X_mar_,
                              np.ndarray[np.float64_t, ndim=1] y_mar_,
                              np.ndarray[np.float64_t, ndim=2] joint):
    cdef double I = 0
    cdef int xlen = X_mar_.shape[0]
    cdef int ylen = y_mar_.shape[0]

    for i in xrange(xlen):
        for j in xrange(ylen):
            if fabs(joint[i,j]) > 10**(-8):
                I += joint[i,j] * log2(joint[i,j] / (X_mar_[i]*y_mar_[j]))
    return I


from cython.parallel import prange
from cython.view cimport array as cvarray
from cpython cimport array as c_array
from array import array

@cython.boundscheck(False)
@cython.wraparound(False)
def _map_binary_array_par(long[:,:] X):

    cdef int N = X.shape[0]
    cdef int n = X.shape[1]
    cyarr = cvarray(shape=(N,), itemsize=sizeof(int), format="i")
    cdef int[:] Xmap = cyarr

    cdef int[:] Xmapout
    Xmapout = _map_binary_array_par_inner(X, Xmap, N, n)

    return np.array(Xmapout,dtype = 'int64')



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] _map_binary_array_par_inner(long[:,:] X, int[:] Xmap, int N, int n):
     cdef int i
     for i in prange(N, schedule=dynamic, nogil=True):
          Xmap[i] = _map_binary_par(X[i,:], n)
     return Xmap


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _map_binary_par(long[:] x, int n) nogil:

    cdef int tot = 0
    cdef int i

    cdef int p = 1
    for i in range(n):
        if x[i]:
            tot += p
        p = p << 1
    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_specific_info(int label, np.float64_t[:] y_mar_,
                  np.float64_t[:,:] cond_Xy,
                  np.float64_t[:,:] cond_yX,
                  np.float64_t[:,:] joint):

    cdef double Ispec = 0
    cdef int    n = cond_Xy.shape[0]
    cdef double mar = y_mar_[label]
    cdef double c
    cdef double contrib

    for x in range(n):
        if mar > 1e-8:
            c = cond_yX[x, label]
            if c > 1e-8:
                contrib = cond_Xy[x, label] * (log2(1.0 / mar)
                                           - log2(1.0 / c))
                Ispec += contrib
    return Ispec



#---------------------------------------------------------------------
# The functions below are not in use, kept for profiling comparison.

@cython.boundscheck(False)
@cython.wraparound(False)
def _map_binary(np.ndarray[np.int64_t, ndim=1] x):

    cdef int tot = 0
    cdef int i
    cdef int n = x.shape[0]

    for i in range(n):
        if x[i]:
            tot += 1<<i
    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _map_binary_cdef(np.int64_t[:] x):

    cdef int tot = 0
    cdef int i
    cdef int n = x.size

    for i in range(n):
        if x[i]:
            tot += 1<<i
    return tot

@cython.boundscheck(False)
@cython.wraparound(False)
def _map_binary_array(np.int64_t[:,:] X):

    cdef int nsamp = X.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] Xmap
    Xmap = np.zeros(nsamp, dtype = np.int64)
    cdef int i

    for i in range(nsamp):
        x = X[i,:]
        Xmap[i] = _map_binary_cdef(x)

    return Xmap