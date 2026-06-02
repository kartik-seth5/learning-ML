import math 
import numpy as np
import matplotlib as plt


class  Value:
    def __init__(self, data, _children=(), _op=''):  # also contains information about what operations were performed to get to the current value, as well as the values themselves that they were performed on 
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + self.other, (self, other), '+') # now the information about this operation is stored on each call
    
    def __mul__(self, other):
        return Value(self.data * self.other, (self, other), '*')
    
    


a = Value(4)
print(a)