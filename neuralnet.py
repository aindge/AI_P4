import math
import numpy


SIGMA = 0
    
class Neuron(object):
    def __init__(self, _data=None):
        self.data = _data
        self.inputs = [1]

    def output(self):
        result = 0
        for i in self.inputs:
            result += (1-math.exp(-SIGMA * i))/(1 + math.exp(-SIGMA * i))