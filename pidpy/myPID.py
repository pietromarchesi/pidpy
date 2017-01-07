import math
import numpy as np


def binary_vector_to_integer(x):
    '''
    Map vector of binary digits to the integer in binary numeral system.
    '''
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

def cantor_tuple_function(x):

    x = np.array(x)
    for i in range(len(x) - 1):
        # tup[i+1] = tup[i] * tup[i+1]
        x[i + 1] = cantor_pairing_function(x[i], x[i + 1])
    return x[i + 1]


def joint_probability(X, y, binary_features = True, n_feature_vals = None):

    ''' Computes the joint probability table of X and y.

    The function computes a probability table with shape (n_patterns, n_labels).
    The mapping of the feature vectors in X to the integer indices in the table
    is performed with two different methods: if the features are binary, every
    feature vector is mapped to the decimal representation of its binary sequence;
    if the features take on integer non-binary values, the Cantor tuple function
    is used to bijectively pair every feature vector with an integer.

    Parameters
    ----------

    X : numpy ndarray
        An array of feature values with shape (n_samples, n_features)

    y : numpy ndarray
        Array of label values with shape (n_samples,)

    binary_features : boolean, optional
        Boolean specifying whether the features take only binary values.
        Default is binary_features = True

    n_feature_vals : integer, optional
        Integer which specifies the number of different values that features
        can take. If not provided, the function computes the size of the set of
        values that appear in the feature array X.


    Returns
    -------
    joint : numpy ndarray
        Normalized probability table with shape (n_labels, n_patterns)

    '''

    Nsamp     = y.shape[0]
    labels    = list(set(y))
    Nfeatures = X.shape[1]

    if not binary_features:
        map = cantor_tuple_function
        if n_feature_vals is not None:
            a = n_feature_vals
        else:
            a = len(set(X.reshape([X.shape[0]*X.shape[1],])))

    else:
        map = binary_vector_to_integer
        a = 2

    n_configurations = a ** Nfeatures
    if not Nsamp > 3 * n_configurations:
        raise ValueError('The number of data points might not be enough to '
                         'accurately estimate the joint probability table.')

    joint = np.zeros([len(labels), n_configurations])

    for i in xrange(Nsamp):
        #print map(X[i,:])
        joint[y[i], map(X[i,:])] += 1

    return joint / float(y.shape[0])


def marginals_from_joint(joint):
    return joint.sum(axis=0), joint.sum(axis=1)

def conditional_probability_from_joint(joint):
    px, py = marginals_from_joint(joint)
    pxy = joint.astype(float) / py[:, np.newaxis]
    pyx = joint.astype(float) / px[np.newaxis, :]
    return pxy, pyx

def mutual_info_from_joint(joint):

    px, py = joint.sum(axis=0), joint.sum(axis=1)
    I = 0
    for i in xrange(py.shape[0]):
        for j in xrange(px.shape[0]):
            if abs(joint[i,j]) > 10**(-8):
                I += joint[i,j] * np.log2(joint[i,j]/ float(py[i]*px[j]))
    return I


def mutual_info_from_joint_test(joint):
    # using the conditional instead of the joint
    px, py = joint.sum(axis=0), joint.sum(axis=1)
    cond_xy, cond_yx = conditional_probability_from_joint(joint)
    I = 0
    for i in xrange(py.shape[0]):
        for j in xrange(px.shape[0]):
            if abs(joint[i,j]) > 10**(-8):
                I += joint[i,j] * np.log2(cond_xy[i,j]/ px[j])
    return I

def mutual_info(X, y, binary_features = True, n_feature_vals = None):
    joint = joint_probability(X, y, binary_features=binary_features,
                      n_feature_vals=n_feature_vals)

    I = mutual_info_from_joint(joint)
    return I

def specific_info_from_joint(joint):

    Ispec = np.zeros(joint.shape[0])
    cond_as, cond_sa = conditional_probability_from_joint(joint)

    for s in xrange(joint.shape[0]):
        for a in xrange(joint.shape[1]):
            ps = joint.sum(axis = 1)[s]
            contribution = cond_as[s,a] * (np.log2(1.0 / ps) - np.log2(1.0 / cond_sa[s,a]))
            if np.isnan(contribution):
                contribution = 0
            Ispec[s] += contribution
    return Ispec


def mutual_info_from_specific_info(joint):
    return np.mean(specific_info_from_joint(joint))



def redundancy(X):

    # compute the joint for every variable with respect to the labels
    # joint_i = joint_probability(X[:,i],y)
    # save them
    # specific_info_from_joint(joint_i)
    # put them together in matrix (s, Ai)
    # ps is simply ps = joint.sum(axis=1)




if False:

    X = np.random.randint(0,3,size = [500,4])

    y = np.random.randint(0,2,size = [500,])

    joint = joint_probability(X, y)
    mi = mutual_info_from_joint(joint)
    Ispec = specific_info_from_joint(joint)
    np.mean(Ispec)


"""
TEST: averaging the specific information over outcomes should give
mutual information.
"""

joint = joint_probability(X,y)
px, py = joint.sum(axis=0), joint.sum(axis=1)
joint = np.ones([2,6])
py = np.array([1,2])
pxy = joint.astype(float) / py[:,np.newaxis]

px = np.array([1,2,3,4,5,6])
pyx = joint.astype(float) / px[np.newaxis,:]


import numpy as np
0.4 * np.log2(2) + 0.1 * np.log2(0.1/(0.5*0.6)) + 0.5 * np.log2(1/0.6)
0.5 * np.log2(2) + 0.5 * np.log2(2)
