import numpy as np
import math
def numerical_derivative_2d(func, epsilon):
    def grad_func(x):
        x0=x[0]
        y0=x[1]
        dx=(func(np.array([x0+epsilon, y0]))-func(np.array([x0, y0])))/epsilon
        dy = (func(np.array([x0, y0+ epsilon])) - func(np.array([x0, y0]))) / epsilon
        return dx+dy
    return grad_func


def grad_descent_2d(func, low, high, start=None, callback=None):
    low=np.array([low, low])
    high= np.array([high, high])
    """ 
    Реализация градиентного спуска для функций двух переменных 
    с несколькими локальным минимумами, но известной квадратной окрестностью
    глобального минимума. Все тесты будут иметь такую природу.



    :param func: np.ndarray -> float — функция 
    :param low: левая граница интервала по каждой из осей
    :param high: правая граница интервала по каждой из осей
    """
    eps = 1e-10
    df = numerical_derivative_2d(func, eps)


    def find_local_min(func, df, low_local, high_local, iters=10000, lr=0.05):
        # функция для нахождения минимума функции f на промежутке (low_local, high_local)
        x= np.random.uniform(low_local[0], high_local[0])
        y = np.random.uniform(low_local[1], high_local[1])
        for i in range(iters):
            # YOUR CODE. Don't forget to clip x to [low_local, high_local]
            x = x - lr * df([x, y])/(math.sqrt(iters))
            y = y - lr * df([x, y]) / (math.sqrt(iters))
            x=np.clip(x, low_local[0], high_local[0])
            x = float(x)
            y = np.clip(y, low_local[1], high_local[1])
            y = float(y)
        return np.array([x, y])

    return find_local_min(func, df, low, high)

def func(x):
    return x[0]**2+x[1]**2


a=grad_descent_2d(func, -5, 5, start=None, callback=None)
print(a)
print(func(a))