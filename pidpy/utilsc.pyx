import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport log2, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_joint_probability(np.ndarray[np.int64_t, ndim=1] X,
                               np.ndarray[np.int64_t, ndim=1] y):

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

@cython.boundscheck(False)
@cython.wraparound(False)
def _map_binary(np.ndarray[np.int64_t, ndim=1] x):

    cdef int tot = 0
    cdef int i
    cdef int n = x.shape[0]

    for i in xrange(n):
        if x[i]:
            tot += 2**i
    return tot

