import numpy as np
import math
def numerical_derivative_2d(func, epsilon):
    def grad_func(x):
        dx=(func(np.array([x[0]+epsilon, x[1]]))-func(x))/epsilon
        dy = (func(np.array([x[0], x[1]+epsilon])) - func(x))/ epsilon
        return np.array([dx, dy])
    return grad_func

def func(x):
    return x[0]**2+x[1]**2


a=numerical_derivative_2d(func, 0.0000001)
print(a)
print(a([1, 2]))