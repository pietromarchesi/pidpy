import numpy as np
import cProfile
import re
import pstats

from pidpy.PIDCalculator import PIDCalculator
from pidpy.PIDCalculatorPP import PIDCalculatorPP


Nsamp = 3000
Nneurons = 4
X = np.random.randint(2,size = [Nsamp,Nneurons])
y = np.random.randint(5,size = Nsamp)

pid = PIDCalculator(X,y)
cProfile.run('pid.synergy()','profiling/synergy_profile')

pid = PIDCalculatorPP(X,y)
cProfile.run('pid.synergy()','profiling/synergy_profile_PP')


p = pstats.Stats('profiling/synergy_profile')
p.sort_stats('tottime').print_stats(10)

p2 = pstats.Stats('profiling/synergy_profile_PP')
p2.sort_stats('tottime').print_stats(10)

