import unittest
import numpy as np

from pidpy import PIDCalculator

class test_LabelMapping(unittest.TestCase):

    def test_LabelMapping_example1(self):
        X = np.random.randint(2,size = [1000,4])
        y = np.random.randint(2,5,size = 1000)

        pid = PIDCalculator(X,y)
        # check that the label vector has been mapped correctly
        np.testing.assert_array_equal(list(set(pid.y)), [0,1,2])

        np.testing.assert_array_equal(pid.labels, [0,1,2])

        #pid = PIDCalculator(X,y,labels = [2,3,4])

        #pid = PIDCalculator(X,y, safe_labels = False)

    # repeat the first 4 tests from Timme et al with different
    # label indices

    def test_Timme2014_Example1(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([1, 4, 4, 1])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 1)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 0)


    def test_Timme2014_Example2(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([3, 4, 3, 4])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 0)
        np.testing.assert_array_almost_equal(uni, [1, 0])
        np.testing.assert_almost_equal(red, 0)


    def test_Timme2014_Example3(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([4, 4, 4, 1])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 0.5)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 0.311, 3)


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

        y = np.array([5, 12, 12, 5, 12, 12, 12, 12, 12, 12])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 0)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 0.0323, 4)