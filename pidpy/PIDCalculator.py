import numpy as np
from pidpy.utils import lazy_property


class PIDCalculator():

    def __init__(self):
        pass

    def initialise(self,X,y):
        self.X = X
        self.y = y
        self.Nsamp = y.shape[0]
        self.Nneurons = X.shape[1]
        self.labels = list(set(y))

    @lazy_property
    def joint_var_(self):
        joint_var_ = joint_var(self.X, self.y)
        return joint_var_

    @lazy_property
    def joint_sub_(self):
        joint_sub_ = joint_sub(self.X, self.y)
        return joint_sub_

    @lazy_property
    def joint_full_(self):
        joint_full_ = joint_full(self.X, self.y)
        return joint_full_

    @lazy_property
    def spec_info_var_(self):
        spec_info_var_ = #compute spec_info_var_
        return spec_info_var_

    @lazy_property
    def spec_info_sub_(self):
        spec_info_sub_ = #compute spec_info_sub_
        return spec_info_sub_

    @lazy_property
    def y_mar_(self):
        y_mar_ = #compute y_mar_
        return y_mar_

    @lazy_property
    def mi_full_(self):
        mi_full_ = #compute mi_full_
        return mi_full_

    def X_mar(self):
        return self.joint_full_.sum(axis=1)

    def y_mar(self):
        return self.joint_full_.sum(axis=0)

    def spec_info_var(self):
        # ugly variable names
        spec_info_var_ = []
        for lab in self.labels:
            spec_info_lab = []
            for i in self.Nneurons:
                info = self.spec_info(lab,self.joint_var_[i])
                spec_info_lab.append(info)
            spec_info_var_.append(spec_info_lab)
        return spec_info_var_

    def spec_info_sub(self):
        spec_info_sub_ = []
        for lab in self.labels:
            spec_info_lab = []
            for i in self.Nneurons:
                info = self.spec_info(lab,self.joint_sub_[i])
                spec_info_lab.append(info)
            spec_info_sub_.append(spec_info_lab)
        return spec_info_sub_

    def redundancy(self):
        self.red = Imin(self.y_mar_, self.spec_info_var_)
        return self.red

    def synergy(self):
        self.syn = self.mi_full_ - Imax(self.y_mar_, self.spec_info_sub_)

    def unique(self):
        uni = np.zeros()

    def surrogate(self):
        sur = PIDCalculator(X, #reshuffled y)
        # and then from him you can get synergy, redundancy, or unique,
        # whatever ya want
        return sur

def joint_probability(X, y):

    if X.ndim > 1:
        Xmap = map_array(X)
    else:
        Xmap = X

    nsamp  = y.shape[0]
    labels = list(set(y))
    vals   = np.array(sorted(list(set(Xmap))))
    nvals  = vals.shape[0]
    joint  = np.zeros([nvals, len(labels)], dtype = int)

    for i in xrange(nsamp):
        joint[np.where(vals == Xmap[i])[0][0], y[i], ] += 1

    return joint / float(y.shape[0])

def map_array(X):
    pass

def joint_full(X,y):
    return joint_probability(X,y)

def joint_var(X,y):
    joints = []
    for i in X.shape[1]:
        joints.append(joint_probability(X[:,i],y))
    # save them in self.joint_individual_
    return joints

def joint_sub(X,y):
    joints = []
    for i in range(X.shape[1]):
        group = group_without_unit(range(X.shape[1]),i)
        joints.append(joint_probability(X[:,group]), y)
    return joints

def spec_info()


def Imin(y_mar_, spec_info):
    Im = 0
    for i in len(y_mar_):
        Im += y_mar_[i]*min(spec_info[i])
    return Im

def Imax(y_mar_, spec_info):
    Im = 0
    for i in len(y_mar_):
        Im += y_mar_[i]*max(spec_info[i])
    return Im

def conditional_probability_from_joint(joint):
    X_mar = joint.sum(axis = 1)
    y_mar = joint.sum(axis = 0)

    cond_Xy = joint.astype(float) / y_mar[:, np.newaxis]
    cond_yX = joint.astype(float) / X_mar[np.newaxis, :]
    return cond_Xy, cond_yX


def specific_info(label, joint):
    y_mar_ = joint.sum(axis = 0) # this you already have
    cond_Xy, cond_yX = conditional_probability_from_joint(joint)

    Ispec = 0
    for x in xrange(cond_Xy.shape[1]):
        contribution = cond_Xy[x,label] * (np.log2(1.0 / y_mar_[label])
                                       - np.log2(1.0 / cond_yX[x,label]))
        if np.isnan(contribution):
            contribution = 0
        Ispec += contribution
    return Ispec