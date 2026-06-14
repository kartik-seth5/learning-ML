import math 
import numpy as np 


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

    def __repr__(self):
        return f"{self.data}"


    def __add__(self, other):
        #on the add operation it creates a new nodes where the value of the node is the sum

        #wrapper for non node types:
        other = other if isinstance(other, Node) else Node(other)       


        out = Node(self.data + other.data, (self, other))
        out.operation = '+'


        def addBackProp():
            #during addition the gradient "trickles through", and is essentially just the gradient of the node above it
            self.grad += out.grad
            other.grad += out.grad
                   
        out.backward = addBackProp 
        
        return out
    
    def __sub__(self, other):
        return self.__add__(self, other * -1)
    
    def __rmul__(self, other):
        return self.__mul__(other)  #is self * other because once a * b is evaluated to not exist python will attempt to run b * a and that is when this function is called
    
    def __mul__(self, other):
        #multiply operaton also creates a new node 

        #wrapper for non node types:
        other = other if isinstance(other, Node) else Node(other)       


        out = Node(self.data * other.data, (self, other))
        out.operation = '*'

        def mulBackProp():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = mulBackProp
        
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        
        assert isinstance(other, (float, int)), "Must be a real number"
        out = Node(self.data**other, (self,))

        out.operation = f'^{other}'

        def powerBackProp():
            self.grad += (other * self.data**(other - 1)) * out.grad
        
        out.backward = powerBackProp

        return out


    def tanh(self):
    
        x = self.data
        _tanh = (np.exp(2 * x) - 1)/(np.exp(2*x) + 1)

        out = Node(_tanh, (self,))

        def tanBackProp():
            self.grad += (1 - _tanh**2) * out.grad
            

        out.operation = 'tanh'
        out.backward = tanBackProp
        return out 
    
    def exp(self):
        x = self.data
        eX = np.exp(self.data)
        out = Node(eX, (self, ))

        def expBackProp():
            self.grad += self.data * eX
        
        out.backward = expBackProp
        out.operation = 'exp'

        return out


a = Node(1.0)

a =  a * 2.0

print(a)