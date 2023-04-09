import numpy as np
import math
def grad_descent_v2(f, df, low=None, high=None, callback=None):
    def find_local_min(f, df, low_local, high_local, iters=5000, lr=0.05):
        # функция для нахождения минимума функции f на промежутке (low_local, high_local)
        x0 = np.random.uniform(low_local, high_local)
        x = x0
        for i in range(iters):
            # YOUR CODE. Don't forget to clip x to [low_local, high_local]
            x = x - lr * df(x)/(math.sqrt(iters))
            if (x < low_local):
                x = low_local
            if (x > high_local):
                x = high_local
            x = float(x)
        return x
    step = np.linspace(low, high, 10)
    arr = []
    arrf = []
    for k in range (15):
        lr=0.01*k
        for i in range(len(step) - 1):
            arr.append(find_local_min(f, df, step[i], step[i + 1]))
            arrf.append(f(arr[i]))
    print(arr)
    print(arrf)
    best_estimate = arr[np.argmin(np.array(arrf))]
    return best_estimate


def F(x):
    return x ** 4 + 5 * x ** 3 - 10 * x
def df(x):
    return 4 * x ** 3 + 15 * x ** 2 - 10
a = grad_descent_v2(F, df, low=-6, high=3)
print(a)