import numpy as np

from pidpy.PIDCalculator import *


X = np.array([[0, 1],
              [0, 1],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0],
              [1, 1],
              [1, 1],
              [1, 1]])

y = np.array([0,1,0,2,2,2,3,2,3])

joint = joint_probability(X,y)
conditional_probability_from_joint(joint)

pidc = PIDCalculator(X,y)

np.testing.assert_array_equal(pidc.joint_var_[0], pidc.joint_sub_[1])
np.testing.assert_array_equal(pidc.joint_var_[1], pidc.joint_sub_[0])

sv = pidc.spec_info_var_
mi = pidc.mi_var_
mifromspec1 = np.sum([pidc.y_mar_[i] * sv[i][0] for i in range(len(sv))])
mifromspec2 = np.sum([pidc.y_mar_[i] * sv[i][1] for i in range(len(sv))])

np.testing.assert_almost_equal(mifromspec1, mi[0])
np.testing.assert_almost_equal(mifromspec2, mi[1])

pidc.synergy()


X = np.array([[0, 0],
              [0, 1],
              [1, 0]])

y = np.array([0,1,2])

pidc = PIDCalculator(X,y)
pidc.synergy()
pidc.unique()
pidc.redundancy()


from pidpy.PIDCalculator import *

X = np.array([[0, 0],
              [0, 1],
              [1, 1],
              [1, 0]])

y = np.array([0,1,1,2])

pidc = PIDCalculator(X,y)
pidc.synergy()
pidc.unique()
pidc.redundancy()

# example 5 from the experimentalist perspective
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

y = np.array([0,1,2,3])

pidc = PIDCalculator(X,y)
print pidc.synergy()
print pidc.unique()
print pidc.redundancy()


# example 1 from the experimentalist perspective
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

y = np.array([0,1,1,0])

pidc = PIDCalculator(X,y)
print pidc.synergy()
print pidc.unique()
print pidc.redundancy()

# example 2 from the experimentalist perspective
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

y = np.array([0,1,0,1])

pidc = PIDCalculator(X,y)
print pidc.synergy()
print pidc.unique()
print pidc.redundancy()


# example 3 from the experimentalist perspective
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])

y = np.array([0,0,0,1])

pidc = PIDCalculator(X,y)
print pidc.synergy()
print pidc.unique()
print pidc.redundancy()


# example 4 from the experimentalist perspective
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

pidc = PIDCalculator(X,y)
print pidc.synergy()
print pidc.unique()
print pidc.redundancy()


from pidpy.PIDCalculator import *

X = np.random.randint(2,size = [16000,11])
y = np.random.randint(10,size = 16000)

pid = PIDCalculator(X,y)
pid.synergy()
pid.redundancy()
pid.unique()


#-------------------------------------------------------

import numpy as np

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

from pidpy import PIDCalculator
pid = PIDCalculator(X,y)
pid.decomposition(20)


#-------------------------------------------------------

import numpy as np
from pidpy.PIDCalculator import *

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

y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1])

pid = PIDCalculator(X, y)
syn = pid.synergy()
uni = pid.unique()
red = pid.redundancy()

np.testing.assert_almost_equal(syn, 0)
np.testing.assert_array_almost_equal(uni, [0, 0])
np.testing.assert_almost_equal(red, 0.0323, 4)

synd = pid.debiased_synergy()
unid = pid.debiased_unique()
redd = pid.debiased_redundancy(n = 10000)

def a(n = 3):
    print n