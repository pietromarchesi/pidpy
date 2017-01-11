import numpy as np
import cProfile
import re
import pstats

from pidpy.PIDCalculator import PIDCalculator

Nsamp = 3000
Nneurons = 5
X = np.random.randint(2,size = [Nsamp,Nneurons])
y = np.random.randint(5,size = Nsamp)

pid = PIDCalculator(X,y)
cProfile.run('pid.synergy()','profiling/synergy_profile')

p = pstats.Stats('profiling/synergy_profile')
p.sort_stats('tottime').print_stats(15)

