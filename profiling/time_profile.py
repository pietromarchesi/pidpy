import numpy as np
import cProfile
import re
import pstats

from pidpy.PIDCalculator import PIDCalculator
from pidpy.PIDCalculatorPP import PIDCalculatorPP


Nsamp = 50000
Nneurons = 10
X = np.random.randint(2,size = [Nsamp,Nneurons])
y = np.random.randint(5,size = Nsamp)

pid = PIDCalculator(X,y)
cProfile.run('pid.synergy()','profiling/synergy_profile')

pid2 = PIDCalculatorPP(X,y)
cProfile.run('pid2.synergy()','profiling/synergy_profile_PP')


p2 = pstats.Stats('profiling/synergy_profile_PP')
p2.sort_stats('tottime').print_stats(10)
p = pstats.Stats('profiling/synergy_profile')
p.sort_stats('tottime').print_stats(10)


%timeit PIDCalculator(X,y).synergy()
%timeit PIDCalculatorPP(X,y).synergy()
