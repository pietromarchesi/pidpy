import numpy as np
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from pidpy.PIDCalculator import PIDCalculator

Nsamp = 3000
Nneurons = 5
X = np.random.randint(2,size = [Nsamp,Nneurons])
y = np.random.randint(5,size = Nsamp)
pid = PIDCalculator(X,y)

with PyCallGraph(output=GraphvizOutput()):
    syn = pid.redundancy()
