import math 
import numpy as np 
import matplotlib as mpl


""" 
How can we create micrograd


we need seperate nodes with different operations at each point


the parent/output node is always guaranteed to have a gradient of 1.0
---> reasonable assumption that we will always know the gradient of the parent node, and that should be used to find the gradient of the child node
-------> this can then be recursed through the CNN

"""


class Node:
    #the basic building block
    def __init__ (self, value=0.0, prev=(),gradient=0.0):
        self.data = value #storing the value at this node 
        self.grad = gradient
        self.children = set(prev) #storing the reference to each of the children nodes
        self.backward = lambda: None #this will be where the backprop function is stored
        self.operation = ''

    def __add__(self, other):
        #on the add operation it creates a new nodes where the value of the node is the sum
       
        out = Node(self.data + other.data, (self, other))
        out.operation = '+'


        def addBackProp():
            #during addition the gradient "trickles through", and is essentially just the gradient of the node above it
            self.grad += out.grad
            other.grad += out.grad
                   
        out.backward = addBackProp 
        
        return out
    
    def __mul__(self, other):
        #multiply operaton also creates a new node 

        out = Node(self.data * other.data, (self, other))
        out.operation = '*'

        def mulBackProp():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = mulBackProp
        
        return out
    
    def tanh(self):
    
        x = self.data
        _tanh = (np.exp(2 * x) - 1)/(np.exp(2*x) + 1, (self))

        def tanBackProp():
            self.grad += (1 - _tanh**2) * out.grad
            
        out = Node(_tanh, (self))
        out.operation = 'tanh'
        out.backward = tanBackProp
        return out 
    

a = Node()
b = Node()
c = Node()
d = Node(3.0)
e = Node(0.4)
f = Node(1.2)
g = Node(2.4)


a = d*e + f*g
b = d * e * f
c.data = 5.0

y = a * b * c

print(y.data)
