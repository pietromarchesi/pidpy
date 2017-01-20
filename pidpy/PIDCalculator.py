import numpy as np
from pidpy.utils import lazy_property
from pidpy.utils import group_without_unit
from pidpy.utils import map_array
from pidpy.utilsc import _compute_mutual_info
from pidpy.utilsc import _compute_joint_probability_bin
from pidpy.utilsc import _compute_joint_probability_nonbin
from pidpy.utilsc import _compute_specific_info


class PIDCalculator():

    '''
    Calculator class for the partial information decomposition of mutual
    information.

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features)
        Data from which to obtain the information decomposition.
    y : 1D ndarray, shape (n_samples, )
        Array containing the labels of the dependent variable.


    Notes
    -----
    Description...

    References
    ----------

    '''


    def __init__(self, *args, **kwargs):

        self.verbosity = 1
        if len(args) == 2:
            self.initialise(args[0], args[1], **kwargs)

    def initialise(self,X,y, safe_labels = False, **kwargs):

        if X.shape[0] != y.shape[0]:
            raise ValueError('The number of samples in the feature and labels'
                             'arrays should match.')

        attributes = ['binary']
        self.__dict__.update((k, v) for k, v in kwargs.iteritems()
                             if k in attributes)

        if not 'labels' in kwargs:
            original_labels = list(set(y))

        else: original_labels = kwargs['labels']

        if not safe_labels:
            if not original_labels == range(len(original_labels)):
                y = np.array([original_labels.index(lab) for lab in y])

        self.labels = range(len(original_labels))
        self.original_labels = original_labels

        self.Nlabels  = len(self.labels)
        self.verbosity = 0
        self.X         = X
        self.y         = y
        self.Nsamp     = y.shape[0]
        self.Nneurons  = X.shape[1]
        self.surrogate_pool = []

        if not hasattr(self, 'binary'):
            self.binary = isbinary(X)

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
        spec_info_var_ = self._spec_info_full(self.labels, self.joint_var_)
        return spec_info_var_

    @lazy_property
    def spec_info_sub_(self):
        spec_info_sub_ = self._spec_info_full(self.labels, self.joint_sub_)
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

    def mutual(self, debiased = False, n = 50, individual = False):
        # TODO implement debiased individual mi
        if debiased:
            mi = self._debiased('mutual', n)
            self.mi = mi
            if individual:
                raise NotImplementedError

        else:
            if individual:
                mi = self.mi_var_
                self.mi_individual = mi
            else:
                mi = self.mi_full_
                self.mi = mi
        return mi


    def redundancy(self, debiased = False, n = 50):
        if debiased:
            self.red = self._debiased('redundancy', n)
        else:
            self.red = Imin(self.y_mar_, self.spec_info_var_)
        return self.red


    def synergy(self, debiased = False, n = 50):
        '''
        Compute the pure synergy between the variables of the data array X.

        Parameters
        ----------
        debiased: bool, optional
            If True, synergy is debiased with shuffled surrogates. Default is
            False.

        n : int, optional
            Number of surrogate data sets to be used for debiasing. Defaults
            to 50.

        Returns
        -------
        synergy : float
            Pure synergy of the variables in `X`.
        standard_deviation : float, optional
            Standard deviation of the synergy of the surrogate sets, only
            returned if `debiased = True`.
        '''

        if debiased:
            self.syn = self._debiased('synergy', n)
        else:
            self.syn = self.mi_full_ - Imax(self.y_mar_, self.spec_info_sub_)
        return self.syn

    def unique(self, debiased = False, n = 50):
        if debiased:
            self.uni = self._debiased('unique', n)
        else:
            uni = np.zeros(self.Nneurons)
            for i in range(self.Nneurons):
                unique = self.mi_var_[i] - Imin(self.y_mar_, self.spec_info_uni_[i])
                uni[i] = unique
            self.uni = uni
        return self.uni

    def _debiased_redundancy(self, n = 50):
        out =  self._debiased('redundancy', n)
        self.debiased_red = out
        return self.debiased_red

    def _debiased_synergy(self, n = 50):

        out = self._debiased('synergy', n)
        self.debiased_syn = out
        return self.debiased_syn

    def _debiased_unique(self, n = 50):
        out = self._debiased('unique', n)
        self.debiased_uni = out
        return self.debiased_uni

    def _debiased_mutual(self, n = 50):
        out = self._debiased('mutual', n)
        self.debiased_mi = out
        return self.debiased_mi

    # TODO: specific info of certain var for all labels
    # TODO: specific info of certain label for all vars
    # TODO: specific info of certain label certain var
    # TODO: the above debiased


    def _debiased(self, fun, n):
        res = getattr(self, fun)()
        self._make_surrogates(n)

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
        if std.shape[0] == 1:
            std = std[0]

        return res - mean, std

    def _make_surrogates(self, n = 50):
        #print('Generating %s surrogates.' %n)
        for i in range(n - len(self.surrogate_pool)):
            self.surrogate_pool.append(self._surrogate())

    def _surrogate(self):
        ind = np.random.permutation(self.Nsamp)
        sur = PIDCalculator(self.X, self.y[ind], verbosity = 1,
                            binary = self.binary, labels = self.labels,
                            safe_labels=True)
        return sur

    def decomposition(self, debiased = False, as_percentage = False, n = 50,
                      decimal = 8, return_individual_unique = False,
                      return_std_surrogates = False):

        # TODO this if statement is very ugly
        if debiased:
            syn, std_syn = self.synergy(debiased = debiased, n = n)
            red, std_red = self.redundancy(debiased = debiased, n = n)
            uni, std_uni = self.unique(debiased = debiased, n = n)
            mi,  std_mi  = self.mutual(debiased = debiased, n = n)

        else:
            syn = self.synergy(debiased = debiased, n = n)
            red = self.redundancy(debiased = debiased, n = n)
            uni = self.unique(debiased = debiased, n = n)
            mi  = self.mutual(debiased = debiased, n = n)

        if as_percentage:
            if mi > 1e-08:
                syn = 100 * syn / mi
                red = 100 * red / mi
                uni = 100 * uni / mi
            else:
                syn, red = 0, 0
                uni = np.zeros_like(uni)

        if not return_individual_unique:
            uni = np.sum(uni)

        ret =  (np.round(syn, decimal), np.round(red, decimal),
                np.round(uni, decimal), np.round(mi, decimal))

        if return_std_surrogates:
            if not return_individual_unique:
                std_uni = np.mean(std_uni)

            std = (std_syn, std_red, std_uni, std_mi)
            ret = (ret, std)

        return ret

    def redundancy_pairs(self):
        red_pairs = []
        for i in range(self.Nneurons):
            for j in range(self.Nneurons):
                if i != j:
                    spec_info_pair = [[self.spec_info_var_[n][i],
                                       self.spec_info_var_[n][j]]
                                       for n in self.labels]

                    red_pair = Imin(self.y_mar_, spec_info_pair)
                    red_pairs.append(red_pair)
        self.red_pairs = np.mean(red_pairs)
        return self.red_pairs

    def _spec_info_full(self, labels, joints):
        spec_info_full_ = []
        for lab in labels:
            spec_info_lab = []
            for joint in joints:
                cond_Xy, cond_yX = conditional_probability_from_joint(joint)
                info = _compute_specific_info(lab, self.y_mar_,
                                      cond_Xy, cond_yX, joint)
                spec_info_lab.append(info)
            spec_info_full_.append(spec_info_lab)
        return spec_info_full_

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



