import numpy as np
from pidpy.utils import lazy_property
from pidpy.utils import group_without_unit
from pidpy.utils import map_array
from pidpy.utilsc import _compute_mutual_info
from pidpy.utilsc import _compute_joint_probability_bin
from pidpy.utilsc import _compute_joint_probability_nonbin


class PIDCalculator():

    def __init__(self, *args):

        self.verbosity = 1
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
        self.binary   = isbinary(X)
        self.surrogate_pool = []
        print('Initialisation successful.')

    @lazy_property
    def joint_var_(self):
        joint_var_ = joint_var(self.X, self.y, binary = self.binary)
        return joint_var_

    @lazy_property
    def joint_sub_(self):
        joint_sub_ = joint_sub(self.X, self.y, binary = self.binary)
        return joint_sub_

    @lazy_property
    def joint_full_(self):
        joint_full_ = joint_probability(self.X, self.y, binary = self.binary)
        return joint_full_

    @lazy_property
    def spec_info_var_(self):
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
        # TODO build this without referring to the full joint
        y_mar_ = self.joint_full_.sum(axis=0)
        return y_mar_

    @lazy_property
    def mi_var_(self):
        mi_var_ = []
        for joint in self.joint_var_:
            x_mar_ = joint.sum(axis = 1)
            mi = _compute_mutual_info(x_mar_, self.y_mar_, joint)
            mi_var_.append(mi)
        return mi_var_

    @lazy_property
    def mi_full_(self):
        mi_full_ = _compute_mutual_info(self.X_mar_, self.y_mar_,
                                       self.joint_full_)
        return mi_full_

    def mutual_info(self):
        return self.mi_full_

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

    def debiased_redundancy(self, n = 50):
        out = self.debiased('redundancy', n)
        self.debiased_syn = out
        return self.debiased_syn

    def debiased_synergy(self, n = 50):
        out = self.debiased('synergy', n)
        self.debiased_red = out
        return self.debiased_red

    def debiased_unique(self, n = 50):
        out = self.debiased('unique', n)
        self.debiased_uni = out
        return self.debiased_uni

    def debiased_mutual_info(self, n = 50):
        out = self.debiased('mutual_info', n)
        self.debiased_mi = out
        return self.debiased_mi

    # def debiased(self,fun, n):
    #     res = getattr(self, fun)()
    #     self.make_surrogates(n)
    #     null = []
    #     for i in range(n):
    #         surrogate = self.surrogate_pool[i]
    #         sval = getattr(surrogate, fun)()
    #         null.append(sval)
    #     if len(null) > 0:
    #         mean = np.mean(null)
    #     else:
    #         mean = 0
    #     std   = np.std(null)
    #     return (res, mean, std)

    def debiased(self,fun, n):
        res = getattr(self, fun)()
        self.make_surrogates(n)

        if fun in ['unique']:
            col = self.Nneurons
        else:
            col = 1

        null = np.zeros([n,col])
        for i in range(n):
            surrogate = self.surrogate_pool[i]
            sval = getattr(surrogate, fun)()
            null[i,:] = sval

        if null.shape[0] > 0:
            mean = np.mean(null, axis = 0)
            std = np.std(null, axis=0)

        else:
            mean = np.zeros(col, dtype = int)
            std = np.empty(col)
            std[:] = np.nan

        if mean.shape[0] == 1:
            mean = mean[0]
        else:
            mean = mean.tolist()

        if std.shape[0] == 1:
            std = std[0]
        else:
            std = std.tolist()

        return (res, mean, std)

    def make_surrogates(self,n = 50):
        for i in range(n - len(self.surrogate_pool)):
            self.surrogate_pool.append(self.surrogate())

    def decomposition(self, n = 50):

        syn = self.debiased_synergy(n)
        red = self.debiased_redundancy(n)
        uni = self.debiased_unique(n)
        mi  = self.debiased_mutual_info(n)
        frac_syn = (syn[0] - syn[1]) / (mi[0] - mi[1])
        frac_red = (red[0] - red[1]) / (mi[0] - mi[1])
        frac_uni = (np.sum(uni[0]) - np.sum(uni[1])) / (mi[0] - mi[1])

        return np.round(frac_syn,4), np.round(frac_red,4), np.round(frac_uni,4)

    def surrogate(self):
        ind = np.random.permutation(self.Nsamp)
        sur = PIDCalculator(self.X, self.y[ind])
        return sur

    # def unique(self):
    #     uni = []
    #     for i in range(self.Nneurons):
    #         unique = self.mi_var_[i] - Imin(self.y_mar_, self.spec_info_sub_)
    #         uni.append(unique)
    #     self.uni = uni
    #     return uni

def isbinary(X):
    return set(X.flatten()) == {0,1}


def joint_probability(X, y, binary = True):
    # TODO _compute_joint_probability_bin can be used
    # selectively when you don't have too many neurons, otherwise
    # you are forced to put in memory huge arrays
    if X.ndim > 1:
        Xmap = map_array(X, binary = binary)
    else:
        Xmap = X

    if binary:
        #nvals = int(''.join(['1' for i in range(10)]), 2)
        joint = _compute_joint_probability_nonbin(Xmap,y)

    else:
        joint = _compute_joint_probability_nonbin(Xmap, y)

    return joint

def joint_var(X,y, binary = True):
    joints = []
    for i in range(X.shape[1]):
        joints.append(joint_probability(X[:,i],y, binary = binary))
    return joints

def joint_sub(X,y, binary = True):
    joints = []
    for i in range(X.shape[1]):
        group = group_without_unit(range(X.shape[1]),i)
        joints.append(joint_probability(X[:,group], y, binary = binary))
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




