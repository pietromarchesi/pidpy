import numpy as np

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

def map(X, nvals):
    # maps X using
    if nvals == 2:
        usemap = map_binary
    else:
        usemap = map_nonbinary

    mapped = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        mapped[i] = usemap(X[i,:])

    return mapped


def joint_probability(x, y):

    #### INCLUDE MAPPING HERE or make bigger function that
    # does mapping and calls this
    nsamp  = y.shape[0]
    labels = list(set(y))
    vals   = np.array(sorted(list(set(x))))
    nvals  = vals.shape[0]
    joint  = np.zeros([len(labels), nvals])

    for i in xrange(nsamp):
        joint[y[i], np.where(vals == x[1])[0][0]] += 1

    return joint / float(y.shape[0])

def mutual_info_from_joint(joint):
    px, py = joint.sum(axis=0), joint.sum(axis=1)
    I = 0
    for i in range(py.shape[0]):
        for j in range(px.shape[0]):
            if abs(joint[i,j]) > 10**(-8):
                I += joint[i,j] * np.log2(joint[i,j]/ float(py[i]*px[j]))
    return I


def conditional_probability_from_joint(joint):
    px, py = marginals_from_joint(joint)
    pxy = joint.astype(float) / py[:, np.newaxis]
    pyx = joint.astype(float) / px[np.newaxis, :]
    return pxy, pyx

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


def joint_individual(X,y):
    joints = []
    for i in X.shape[1]:
        joints.append(joint_probability(X[:,i],y))
    # save them in self.joint_individual_
    return joints

def joint_subgroups(X,y):
    joints = []
    for i in range(X.shape[1]):
        group = group_without_unit(range(X.shape[1]),i)
        joints.append(joint_probability(X[:,group]), y)


def group_without_unit(group, unit):
    """ Returns the tuple given by group without the element give by unit. """
    if isinstance(unit, int):
        unit = [unit]
    return tuple(k for k in group if not k in unit)


def Iminmax(joints):

    # get prior of label - over the full joint??
    # sp_info_min = []
    # sp_info_max = []
    #for every label:
        # spec_infos = []
        # for every joint:
            #spec_infos.append(specific_info(joint))
        # sp_info_min.append(min(spec_infos))
        # sp_info_max.append(max(spec_infos))
    #

