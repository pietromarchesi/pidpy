import numpy as np

from pidpy.PIDCalculatorPP import compute_joint_probability
from pidpy.PIDCalculatorPP import compute_mutual_info
from pidpy.PIDCalculatorPP import joint_probability
from pidpy.utilsc import _compute_joint_probability

X = np.random.randint(8,size = 1000)
y = np.random.randint(10,size = 1000)

jpp = compute_joint_probability(X,y)
jcy =_compute_joint_probability(X,y)
np.testing.assert_array_equal(jpp,jcy)

print('Joint probability - pure Python')
%timeit compute_joint_probability(X,y)
print('Joint probability - Cython')
%timeit _compute_joint_probability(X,y)

X = np.random.randint(5,size = [1000,5])
y = np.random.randint(10,size = 1000)
joint = joint_probability(X,y)

X_mar_ = joint.sum(axis = 1)
y_mar_ = joint.sum(axis = 0)
mipp = compute_mutual_info(X_mar_, y_mar_, joint)
micy = _compute_mutual_info(X_mar_, y_mar_, joint)
np.testing.assert_array_equal(mipp,micy)

print('Mutual information - pure Python')
%timeit compute_mutual_info(X_mar_, y_mar_, joint)
%timeit _compute_mutual_info(X_mar_, y_mar_, joint)
