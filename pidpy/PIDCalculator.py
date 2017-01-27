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
    information for discrete variables.

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features)
        Data from which to obtain the information decomposition.
    y : 1D ndarray, shape (n_samples, )
        Array containing the labels of the dependent variable.


    Notes
    -----
    Description of the method.
    Partial information decomposition of binary data (`X` contains only
    `0` and `1`, no restrictions on `y`) is supported for an arbitrary
    number of variables (although ensure that the number of data points is
    sufficient to accurately build the probability tables).
    Decomposition of integer non-binary data is only supported for up to
    three variables.

    References
    ----------
    '''


    def __init__(self, *args, **kwargs):

        self.verbosity = 1
        if len(args) == 2:
            self._initialise(args[0], args[1], **kwargs)

    def _initialise(self, X, y, safe_labels = False, **kwargs):
        '''
        Initialise the PID calculator.

        Parameters
        ----------
        X : 2D ndarray, shape (n_samples, n_features)
            Data from which to obtain the information decomposition.

        y : 1D ndarray, shape (n_samples, )
            Array containing the labels of the dependent variable.

        safe_labels : bool, optional (default = False)
            If `True`, it is assumed that the `n` labels the compose `y`
            are the integers in `range(n)`. If `False`, the above is checked
            and if it is found to be false, `y` is mapped so that the label
            values are `range(n)`. This is necessary because the label values
            are used for indexing in the construction of probability tables.
            `safe_labels` should be kept to `False`, setting to `True` is only
            done internally to speed up the initialization of the PID calculators
            used to generate surrogate data.
        '''

        if X.shape[0] != y.shape[0]:
            raise ValueError('The number of samples in the feature and labels'
                             'arrays should match.')

        if not issubclass(X.dtype.type, np.integer):
            X = X.astype('int')

        attributes = ['binary']
        self.__dict__.update((k, v) for k, v in kwargs.iteritems()
                             if k in attributes)

        # labels are passed by the main calculator as kwargs to the calculators
        # used for debiasing, to avoid recomputing the set of labels
        if not 'labels' in kwargs:
            original_labels = list(set(y))
        else:
            original_labels = kwargs['labels']

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
        standard_deviation : float
            Standard deviation of the synergy of the surrogate sets, only
            returned if `debiased = True`.
        '''

        if debiased:
            self.syn = self._debiased('synergy', n)
        else:
            self.syn = self.mi_full_ - Imax(self.y_mar_, self.spec_info_sub_)
        return self.syn

    def redundancy(self, debiased = False, n = 50):
        '''
        Compute the pure synergy between the variables of the data array X.

        Parameters
        ----------
        debiased: bool, optional
            If True, redundancy is debiased with shuffled surrogates. Default is
            False.

        n : int, optional
            Number of surrogate data sets to be used for debiasing. Defaults
            to 50.

        Returns
        -------
        redundancy : float
            Pure redundancy of the variables in `X`.
        standard_deviation : float
            Standard deviation of the redundancy of the surrogate sets, only
            returned if `debiased = True`.
        '''

        if debiased:
            self.red = self._debiased('redundancy', n)
        else:
            self.red = Imin(self.y_mar_, self.spec_info_var_)
        return self.red


    def unique(self, debiased = False, n = 50):
        '''
        Compute the unique information of the variables in the data array X.

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
        unique : 1D ndarray (n_variables, )
            Unique information of each of the variables in `X`.
        standard_deviation : 1D ndarray (n_variables, )
            Standard deviation of the unique information of the surrogate sets
            for each variable in `X`.
            Only returned if `debiased = True`.
        '''

        if debiased:
            self.uni = self._debiased('unique', n)
        else:
            uni = np.zeros(self.Nneurons)
            for i in range(self.Nneurons):
                unique = self.mi_var_[i] - Imin(self.y_mar_, self.spec_info_uni_[i])
                uni[i] = unique
            self.uni = uni
        return self.uni

    def mutual(self, debiased = False, n = 50, individual = False, decimals = 8):

        '''
        Compute the mutual between `X` and `y`.

        Parameters
        ----------
        debiased: bool, optional (default = False)
            If True, mutual information is debiased with shuffled surrogates.

        n : int, optional (default = 50)
            Number of surrogate data sets to be used for debiasing.

        individual : bool, optional (default = False)
            If `False`, mutual information between the full `X` and `y`
            is calculated.
            If `True`, mutual information is computed for each of the
            individual variables (columns) which compose `X`, and an array
            of MI values is returned.

        Returns
        -------
        mutual information
            Mutual information in bits. If `debiased = False`, a single
            float value is returned (or a list if `individual = True`).
            If `debiased = True`, a tuple is returned where the first element
            contains the MI values (float or 1D ndarray of floats,
            depending on `individual`) and the second
            element contains the associated standard deviation(s).

        Examples
        --------

        >>> X = np.array([[0, 0],
                          [1, 0],
                          [0, 1],
                          [1, 1]])
        >>> y = np.array([0, 1, 1, 0])
        >>> pid = PIDCalculator(X,y)
        >>> pid.mutual()
        1.0
        >>> pid.mutual(individual=True)
        [0.0, 0.0]
        '''
        if debiased:
            mi = self._debiased('mutual', n, individual = False)
            self.mi = mi
            if individual:
                mi = self._debiased('mutual',n, individual = True)

        else:
            if individual:
                mi = self.mi_var_
                self.mi_individual = mi
            else:
                mi = self.mi_full_
                self.mi = mi

        return mi

    def decomposition(self, debiased = False, as_percentage = False, n = 50,
                      decimal = 8, return_individual_unique = False,
                      return_std_surrogates = False):
        '''
        Decompose the mutual information of `X` into the pure synergy,
        redunandcy, and unique terms.

        Parameters
        ----------
        debiased : bool, optional (default = False)
            If True, the values of the decomposition are debiased using
            shuffled surrogates.

        as_percentage : bool, optional (default = False)
            If True, the values of synergy, redundancy, and mutual information
            are returned as a percentage of mutual information. The mutual
            information value is returned in bits.

        n : int, optional (default = 50)
            Number of shuffled data sets used for debiasing.

        decimal : int, optional (default = 8)
            Round off the output to a certain decimal position.

        return_individual_unique : bool, optional (default = False)
            If True, unique information is returned as an array containing
            the corresponding value for each variable in `X`. If False,
            it is returned as the sum of all the individual unique information
            terms.

        return_std_surrogates : bool, optional (default = False)
            If true, the standard deviation associated with the surrogate data
            sets for each information theoretic measure is returned.


        Returns
        -------
        decomposition : tuple
            If `return_std_surrogates = False`, the tuple
            `decomposition = (synergy, redundancy, unique, mutual)` is returned
            (with the values of synergy, redundancy, and unique information
            expressed in bits if `as_percentage = False`, or as
            percentages of the mutual information if `as_percentage = True`).
            If `return_std_surrogates = True`, decomposition is a tuple containing
            two tuples, namely
            `decomposition = ((synergy, redundancy, unique, mutual),
            (std_synergy, std_redundancy, std_unique, std_mutual))`

        Examples
        --------
        Compute the partial information decomposition
        >>> pid = PIDCalculator(X,y)
        >>> syn, red, uni, mut = pid.decomposition()

        Compute the partial information decomposition with 100 surrogates
        used for debiasing, and return the decomposition terms as a percentage
        of mutual information (the latter is still returned in bits)
        >>> syn, red, uni, mut = pid.decomposition(debiased = True, n = 30, as_percentage=True)
        '''

        if debiased:
            syn, std_syn = self.synergy(debiased = True, n = n)
            red, std_red = self.redundancy(debiased = True, n = n)
            uni, std_uni = self.unique(debiased = True, n = n)
            mi,  std_mi  = self.mutual(debiased = True, n = n)

        else:
            syn = self.synergy(debiased = False)
            red = self.redundancy(debiased = False)
            uni = self.unique(debiased = False)
            mi  = self.mutual(debiased = False)

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

        decomposition =  (np.round(syn, decimal), np.round(red, decimal),
                np.round(uni, decimal), np.round(mi, decimal))

        if return_std_surrogates:
            if not return_individual_unique:
                std_uni = np.mean(std_uni)

            if as_percentage:
                if mi > 1e-08:
                    std_syn = 100 * std_syn / mi
                    std_red = 100 * std_red / mi
                    std_uni = 100 * std_uni / mi

            std_decomposition = (std_syn, std_red, std_uni, std_mi)

            decomposition = (decomposition, std_decomposition)

        return decomposition

    def _debiased_synergy(self, n = 50):
        out = self._debiased('synergy', n)
        self.debiased_syn = out
        return self.debiased_syn

    def _debiased_redundancy(self, n = 50):
        out =  self._debiased('redundancy', n)
        self.debiased_red = out
        return self.debiased_red

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

    def _debiased(self, fun, n, **kwargs):
        res = getattr(self, fun)(**kwargs)
        self._make_surrogates(n)

        if fun in ['unique'] or \
                ('individual' in kwargs and kwargs['individual']):
            col = self.Nneurons
        else:
            col = 1

        null = np.zeros([n,col])
        for i in range(n):
            surrogate = self.surrogate_pool[i]
            sval = getattr(surrogate, fun)(**kwargs)
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



    def redundancy_pairs(self):
        '''
        Experimental function to compute the average redundancy between
         all the pairs of variables in `X`.
        '''
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



