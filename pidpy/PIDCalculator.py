import numpy as np
from pidpy.utils import lazy_property
from pidpy.utils import group_without_unit
from pidpy.utils import map_array

class PIDCalculator():

    def __init__(self, *args):
        if len(args) == 2:
            self.initialise(args[0], args[1])

    def initialise(self,X,y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('The number of samples in the feature and labels'
                             'arrays should match.')
        self.X        = X
        self.y        = y
        self.Nsamp    = y.shape[0]
        self.Nneurons = X.shape[1]
        self.labels   = list(set(y))
        self.Nlabels  = len(self.labels)
        print('Initialisation successful.')

    @lazy_property
    def joint_var_(self):
        print('Computing joint probability tables for each '
              'individual input variable.')
        joint_var_ = joint_var(self.X, self.y)
        return joint_var_

    @lazy_property
    def joint_sub_(self):
        print('Computing joint probability tables for all groups'
              'of n-1 variables.')
        joint_sub_ = joint_sub(self.X, self.y)
        return joint_sub_

    @lazy_property
    def joint_full_(self):
        print('Computing the full probability table.')
        joint_full_ = joint_probability(self.X, self.y)
        return joint_full_

    @lazy_property
    def spec_info_var_(self):
        print('Computing specific information for each individual'
              'input variable')
        spec_info_var_ = spec_info_full(self.labels, self.joint_var_)
        return spec_info_var_

    @lazy_property
    def spec_info_sub_(self):
        spec_info_sub_ = spec_info_full(self.labels, self.joint_sub_)
        return spec_info_sub_

    @lazy_property
    def spec_info_uni_(self):
        spec_info_uni_ = []
        sv = self.spec_info_var_
        ss = self.spec_info_sub_
        for i in range(self.Nneurons):
            spec_info_i = [[sv[j][i], ss[j][i]] for j in range(self.Nlabels)]
            spec_info_uni_.append(spec_info_i)
        return spec_info_uni_

    @lazy_property
    def X_mar_(self):
        X_mar_ = self.joint_full_.sum(axis=1)
        return X_mar_

    @lazy_property
    def y_mar_(self):
        y_mar_ = self.joint_full_.sum(axis=0)
        return y_mar_

    @lazy_property
    def mi_var_(self):
        mi_var_ = []
        for joint in self.joint_var_:
            x_mar_ = joint.sum(axis = 1)
            mi = compute_mutual_info(x_mar_, self.y_mar_, joint)
            mi_var_.append(mi)
        return mi_var_

    @lazy_property
    def mi_full_(self):
        mi_full_ = compute_mutual_info(self.X_mar_, self.y_mar_,
                                       self.joint_full_)
        return mi_full_


    def redundancy(self):
        self.red = Imin(self.y_mar_, self.spec_info_var_)
        return self.red

    def synergy(self):
        self.syn = self.mi_full_ - Imax(self.y_mar_, self.spec_info_sub_)
        return self.syn

    def unique(self):
        uni = []
        for i in range(self.Nneurons):
            unique = self.mi_var_[i] - Imin(self.y_mar_, self.spec_info_uni_[i])
            uni.append(unique)
        self.uni = uni
        return self.uni

    # def unique(self):
    #     uni = []
    #     for i in range(self.Nneurons):
    #         unique = self.mi_var_[i] - Imin(self.y_mar_, self.spec_info_sub_)
    #         uni.append(unique)
    #     self.uni = uni
    #     return uni

    def surrogate(self):
        ind = np.random.permutation(self.Nsamp)
        sur = PIDCalculator(self.X, self.y[ind])
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



def joint_var(X,y):
    joints = []
    for i in range(X.shape[1]):
        joints.append(joint_probability(X[:,i],y))
    return joints

def joint_sub(X,y):
    joints = []
    for i in range(X.shape[1]):
        group = group_without_unit(range(X.shape[1]),i)
        joints.append(joint_probability(X[:,group], y))
    return joints

def Imin(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i]*np.min(spec_info[i])
    return Im

def Imax(y_mar_, spec_info):
    Im = 0
    for i in range(len(y_mar_)):
        Im += y_mar_[i]*np.max(spec_info[i])
    return Im

def conditional_probability_from_joint(joint):
    X_mar = joint.sum(axis = 1)
    y_mar = joint.sum(axis = 0)

    cond_Xy = joint.astype(float) / y_mar[np.newaxis, :]
    cond_yX = joint.astype(float) / X_mar[:, np.newaxis]
    return cond_Xy, cond_yX

def spec_info_full(labels, joints):
    spec_info_full_ = []
    for lab in labels:
        spec_info_lab = []
        for joint in joints:
            info = specific_info(lab, joint)
            spec_info_lab.append(info)
        spec_info_full_.append(spec_info_lab)
    return spec_info_full_

def specific_info(label, joint):
    y_mar_ = joint.sum(axis = 0) # this you already have
    cond_Xy, cond_yX = conditional_probability_from_joint(joint)

    Ispec = 0
    for x in xrange(cond_Xy.shape[0]):
        contribution = cond_Xy[x,label] * (np.log2(1.0 / y_mar_[label])
                                       - np.log2(1.0 / cond_yX[x,label]))
        if np.isnan(contribution):
            contribution = 0
        Ispec += contribution
    return Ispec


def compute_mutual_info(X_mar_, y_mar_, joint):
    I = 0
    for i in range(X_mar_.shape[0]):
        for j in range(y_mar_.shape[0]):
            if abs(joint[i,j]) > 10**(-8):
                I += joint[i,j] * np.log2(joint[i,j]/ float(X_mar_[i]*y_mar_[j]))
    return I
