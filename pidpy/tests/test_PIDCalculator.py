import unittest
import numpy as np

from pidpy import PIDCalculator


def compare_values(X, y, test_syn=None, test_red=None,
                   test_uni=None, test_mi=None, test_mi_vars=None,
                   decimal=3):
    '''
    Utility function to test pidpy results against examples in the literature.
    '''

    pid = PIDCalculator(X, y)
    syn = pid.synergy()
    uni = pid.unique()
    red = pid.redundancy()
    mi = pid.mutual()

    syn_d, syn_std = pid.synergy(debiased=True, n=0)
    uni_d, uni_std = pid.unique(debiased=True, n=0)
    red_d, red_std = pid.redundancy(debiased=True, n=0)
    mi_d, mi_std = pid.mutual(debiased=True, n=0)

    mi_vars = pid.mutual(individual=True)
    mi_vars2 = pid.mutual(debiased = True, individual=True, n = 0)[0]


    dec = pid.decomposition(debiased=False, as_percentage=False,
                            return_individual_unique=True)

    if test_syn is not None:
        np.testing.assert_almost_equal(syn, test_syn, decimal=decimal,
                                       err_msg='Synergy mismatch')
        np.testing.assert_almost_equal(dec[0], test_syn, decimal=decimal,
                                       err_msg='Synergy [decomposition] mismatch')
        np.testing.assert_almost_equal(syn_d, test_syn, decimal=decimal,
                                       err_msg='Synergy [debiased] mismatch')
    if test_red is not None:
        np.testing.assert_almost_equal(red, test_red, decimal=decimal,
                                       err_msg='Redundancy mismatch')
        np.testing.assert_almost_equal(dec[1], test_red, decimal=decimal,
                                       err_msg='Redundancy [decomposition] mismatch')
        np.testing.assert_almost_equal(red_d, test_red, decimal=decimal,
                                       err_msg='Redundancy [debiased] mismatch')
    if test_uni is not None:
        np.testing.assert_array_almost_equal(uni, test_uni, decimal=decimal,
                                             err_msg='Unique Info mismatch')
        np.testing.assert_array_almost_equal(dec[2], test_uni,
                                             decimal=decimal,
                                             err_msg='Unique [decomposition] mismatch')
        np.testing.assert_array_almost_equal(uni_d, test_uni,
                                             decimal=decimal,
                                             err_msg='Unique [debiased] mismatch')
    if test_mi is not None:
        np.testing.assert_almost_equal(mi, test_mi, decimal=decimal,
                                       err_msg='Mutual Info (global) mismatch')
        np.testing.assert_array_almost_equal(dec[3], test_mi,
                                             decimal=decimal,
                                             err_msg='Mutual [decomposition] mismatch')
        np.testing.assert_array_almost_equal(mi_d, test_mi,
                                             decimal=decimal,
                                             err_msg='Mutual [debiased] mismatch')
    if test_mi_vars is not None:
        np.testing.assert_array_almost_equal(mi_vars, test_mi_vars,
                                             decimal=decimal,
                                             err_msg='Mutual Info (individual) mismatch')

        np.testing.assert_array_almost_equal(mi_vars2, test_mi_vars,
                                             decimal=decimal,
                                             err_msg='Mutual Info (individual) mismatch')


class test_ExampleGatesTimme2014(unittest.TestCase):
    '''
    Examples gates taken from:

    Timme, Nicholas, et al. "Synergy, redundancy, and multivariate information
    measures: an experimentalist's perspective." Journal of computational
    neuroscience 36.2 (2014): 119-140.
    '''

    def test_Timme2014_Example1(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 1, 1, 0])

        test_syn     = 1
        test_red     = 0
        test_uni     = [0, 0]
        test_mi      = 1
        test_mi_vars = [0, 0]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars)

    def test_Timme2014_Example2(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 1, 0, 1])

        test_syn     = 0
        test_red     = 0
        test_uni     = [1, 0]
        test_mi      = 1
        test_mi_vars = [1, 0]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars)

    def test_Timme2014_Example3(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 0, 0, 1])


        test_syn     = 0.5
        test_red     = 0.311
        test_uni     = [0, 0]
        test_mi      = 0.811
        test_mi_vars = [0.311, 0.311]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars)

    def test_Timme2014_Example4(self):
        X = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1]])

        y = np.array([0,1,1,0,1,1,1,1,1,1])

        test_syn     = 0
        test_red     = 0.0323
        test_uni     = [0, 0]
        test_mi      = 0.0323
        test_mi_vars = [0.0323, 0.0323]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars, decimal = 4)

    def test_Timme2014_Example5(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 1, 2, 3])

        test_syn     = 1
        test_red     = 1
        test_uni     = [0, 0]
        test_mi      = 2
        test_mi_vars = [1, 1]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars, decimal = 4)

    def test_Timme2014_Example6(self):

        X = np.array([[0, 0],
                      [1, 1]])

        y = np.array([0, 0])

        test_syn     = 0
        test_red     = 0
        test_uni     = [0, 0]
        test_mi      = 0
        test_mi_vars = [0, 0]

        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars, decimal = 7)

    def test_Timme2014_Example7(self):

        X = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]])

        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])

        test_syn     = 1
        test_mi      = 1
        test_mi_vars = [0, 0, 0]

        compare_values(X,y,test_syn = test_syn,
                                test_red = None,
                                test_uni = None,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars, decimal = 4)

        test_syn     = 0
        compare_values(X[:,[0,1]],y,test_syn = test_syn)


    def test_Timme2014_Example8(self):

        X = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]])

        y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

        test_syn     = 0
        test_mi      = 1
        test_mi_vars = [0, 0, 0]

        compare_values(X,y,test_syn = test_syn,
                                test_red = None,
                                test_uni = None,
                                test_mi  = test_mi,
                                test_mi_vars = test_mi_vars, decimal = 4)

        test_syn     = 1
        compare_values(X[:,[0,1]],y,test_syn = test_syn)

    def test_Timme2014_Example4_testLabels(self):
        # test that label mapping to different integers gives the same
        # result

        X = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1],
                      [1, 1]])

        y = np.array([12, 31, 31, 12, 31, 31, 31, 31, 31, 31])

        test_syn = 0
        test_red = 0.0323
        test_uni = [0, 0]
        test_mi = 0.0323
        test_mi_vars = [0.0323, 0.0323]

        compare_values(X, y, test_syn=test_syn,
                            test_red=test_red,
                            test_uni=test_uni,
                            test_mi=test_mi,
                            test_mi_vars=test_mi_vars, decimal=4)

    def test_Timme2014_Example8_testLabels(self):

        X = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]])

        y = np.array([2, 8, 8, 2, 2, 8, 8, 2])

        test_syn = 0
        test_mi = 1
        test_mi_vars = [0, 0, 0]

        compare_values(X, y, test_syn=test_syn,
                            test_red=None,
                            test_uni=None,
                            test_mi=test_mi,
                            test_mi_vars=test_mi_vars, decimal=4)

        test_syn = 1
        compare_values(X[:, [0, 1]], y, test_syn=test_syn)




class test_ExampleGatesInce2016(unittest.TestCase):
    '''
    Examples taken from:

    Williams, Paul L., and Randall D. Beer.
    "Nonnegative decomposition of multivariate information."
    arXiv preprint arXiv:1004.2515 (2010).
    '''

    def test_Ince2016_Figure3A(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0]])

        y = np.array([0, 1, 2])


        test_syn     = 0.3333
        test_red     = 0.5850
        test_uni     = [0.3333, 0.3333]


        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                decimal  = 4)


    def test_Ince2016_Figure3B(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0]])

        y = np.array([0, 1, 1, 2])


        test_syn     = 0.5
        test_red     = 0.5
        test_uni     = [0, 0.5]


        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                decimal  = 7)


    def test_Ince2016_Figure3C(self):
        X = np.array([[0, 0],
                      [1, 1],
                      [0, 1],
                      [1, 1],
                      [0, 1],
                      [1, 0]])

        y = np.array([0, 0, 1, 1, 2, 2])


        test_syn     = 0.6667
        test_red     = 0
        test_uni     = [0, 0.2516]


        compare_values(X,y,test_syn = test_syn,
                                test_red = test_red,
                                test_uni = test_uni,
                                decimal  = 4)


    class test_DebiasedMeasures(unittest.TestCase):

        # TODO: tile the arrays from the examples and test debiased measures
        # maybe introducing some noise?
        # make sure that if you run debiased on repetitions, you get
        # very close to the true value


