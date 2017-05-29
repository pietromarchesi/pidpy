import numpy as np
from pidpy import PIDCalculator

X = np.random.randint(2,size=[30,3])
y = np.random.randint(10,size=[30,])

pid = PIDCalculator(X, y)

pid.decomposition(debiased=True, return_std_surrogates=True, test_significance=True)


pid.decomposition(debiased=True, return_std_surrogates=False, test_significance=True)