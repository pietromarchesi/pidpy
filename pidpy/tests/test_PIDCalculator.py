import unittest
import numpy as np

from pidpy import PIDCalculator


class test_ExampleGatesTimme2014(unittest.TestCase):

    def test_Timme2014_Example1(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 1, 1, 0])

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

        y = np.array([0, 1, 0, 1])

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

        y = np.array([0, 0, 0, 1])

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

        y = np.array([0,1,1,0,1,1,1,1,1,1])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 0)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 0.0323, 4)

    def test_Timme2014_Example5(self):
        X = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])

        y = np.array([0, 1, 2, 3])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 1)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 1)

    def test_Timme2014_Example6(self):

        X = np.array([[0, 0],
                      [1, 1]])

        y = np.array([0, 0])

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        uni = pid.unique()
        red = pid.redundancy()

        np.testing.assert_almost_equal(syn, 0)
        np.testing.assert_array_almost_equal(uni, [0, 0])
        np.testing.assert_almost_equal(red, 0)

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

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        np.testing.assert_almost_equal(syn, 1)

        pid = PIDCalculator(X[:,[0,1]],y)
        syn = pid.synergy()
        np.testing.assert_almost_equal(syn, 0)


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

        pid = PIDCalculator(X, y)
        syn = pid.synergy()
        np.testing.assert_almost_equal(syn, 0)

        pid = PIDCalculator(X[:,[0,1]],y)
        syn = pid.synergy()
        np.testing.assert_almost_equal(syn, 1)

#
# from pidpy.PIDCalculator import *
#
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 1],
#               [1, 0]])
#
# y = np.array([0,1,1,2])
#
# pidc = PIDCalculator(X,y)
# pidc.synergy()
# pidc.unique()
# pidc.redundancy()
#
# # example 5 from the experimentalist perspective
# X = np.array([[0, 0],
#               [1, 0],
#               [0, 1],
#               [1, 1]])
#
# y = np.array([0,1,2,3])
#
# pidc = PIDCalculator(X,y)
# print pidc.synergy()
# print pidc.unique()
# print pidc.redundancy()
#
#
# # example 1 from the experimentalist perspective
# X = np.array([[0, 0],
#               [1, 0],
#               [0, 1],
#               [1, 1]])
#
# y = np.array([0,1,1,0])
#
# pidc = PIDCalculator(X,y)
# print pidc.synergy()
# print pidc.unique()
# print pidc.redundancy()
#
# # example 2 from the experimentalist perspective
# X = np.array([[0, 0],
#               [1, 0],
#               [0, 1],
#               [1, 1]])
#
# y = np.array([0,1,0,1])
#
# pidc = PIDCalculator(X,y)
# print pidc.synergy()
# print pidc.unique()
# print pidc.redundancy()
#
#
# # example 3 from the experimentalist perspective
# X = np.array([[0, 0],
#               [1, 0],
#               [0, 1],
#               [1, 1]])
#
# y = np.array([0,0,0,1])
#
# pidc = PIDCalculator(X,y)
# print pidc.synergy()
# print pidc.unique()
# print pidc.redundancy()
#
#
# # example 4 from the experimentalist perspective
# X = np.array([[0, 0],
#               [0, 0],
#               [0, 0],
#               [1, 1],
#               [1, 1],
#               [1, 1],
#               [1, 1],
#               [1, 1],
#               [1, 1],
#               [1, 1]])
#
# y = np.array([0,1,1,0,1,1,1,1,1,1])
#
# pidc = PIDCalculator(X,y)
# print pidc.synergy()
# print pidc.unique()
# print pidc.redundancy()
#
# class test_NAME(unittest.TestCase):
#
#     def setUp(self):
#         print('setting up!')
#
#     def test_feature1(self):
#         print('testing feature 1!')
#
#     def test_feature2(self):
#         print('testing feature 2!')
#
#     def test_fail(self):
#         self.assertEqual(3,1)
#
#     def tearDown(self):
#         print('we are done here!')
#
#
#
# class test_MutualSpecificInformation
# X = np.array([[0, 1],
#               [0, 1],
#               [0, 1],
#               [1, 0],
#               [1, 0],
#               [1, 0],
#               [1, 1],
#               [1, 1],
#               [1, 1]])
#
# y = np.array([0,1,0,2,2,2,3,2,3])
#
# joint = joint_probability(X,y)
# conditional_probability_from_joint(joint)
#
# pidc = PIDCalculator(X,y)
#
# np.testing.assert_array_equal(pidc.joint_var_[0], pidc.joint_sub_[1])
# np.testing.assert_array_equal(pidc.joint_var_[1], pidc.joint_sub_[0])
#
# sv = pidc.spec_info_var_
# mi = pidc.mi_var_
# mifromspec1 = np.sum([pidc.y_mar_[i] * sv[i][0] for i in range(len(sv))])
# mifromspec2 = np.sum([pidc.y_mar_[i] * sv[i][1] for i in range(len(sv))])
#
# np.testing.assert_almost_equal(mifromspec1, mi[0])
# np.testing.assert_almost_equal(mifromspec2, mi[1])
#
# pidc.synergy()
