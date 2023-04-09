import numpy as np
def grad_descent_v1(f, deriv, x0=None, lr=0.1, iters=100, callback=None):
    if x0 is None:
        x0 = np.random.uniform()

    x = x0
    print(type(x))
    callback(x, f(x))  # не забывайте логировать

    for i in range(0, iters):
        x = x - lr * deriv(x)

    return x
def F(x):
    return x**4+5*x**3-10*x
def df(x):
    return 4*x**3+15*x**2-10
a=grad_descent_v1(F, df, x0=None, lr=0.1, iters=100, callback=None)
print(a)