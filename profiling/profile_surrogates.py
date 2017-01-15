import numpy as np
import cProfile
import re
import pstats

from pidpy.PIDCalculator import PIDCalculator


Nsamp = 6000
Nneurons = 10
X = np.random.randint(2,size = [Nsamp,Nneurons])
y = np.random.randint(5,size = Nsamp)

pid = PIDCalculator(X,y)
cProfile.run('pid.debiased_synergy(30)','profiling/deb_syn_profile_opt')


p = pstats.Stats('profiling/deb_syn_profile_noopt')
p.sort_stats('tottime').print_stats(10)

p = pstats.Stats('profiling/deb_syn_profile_opt')
p.sort_stats('tottime').print_stats(10)

