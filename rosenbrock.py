import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LB = -5
UB = 10
funcName = "Rosenbrock"


def func(x):
    return np.sum([100*np.square(x[i+1]-np.square(x[i])) + np.square(x[i]-1) for i in range(len(x)-1)])


def generate_points(min_val, max_val, num_points):
    return np.linspace(min_val, max_val, num_points)


if __name__ == "__main__":
    dimension, amount_of_points = 2, 100
    X = generate_points(LB, UB, amount_of_points)
    Y = generate_points(LB, UB, amount_of_points)

    A = [[X[x] for _ in range(amount_of_points)]
         for x in range(amount_of_points)]
    B = [[Y[y] for y in range(amount_of_points)]
         for _ in range(amount_of_points)]
    C = [[func([X[x], Y[y]]) for y in range(amount_of_points)]
         for x in range(amount_of_points)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.array(A), np.array(B), np.array(C), rstride=1, cstride=1, cmap=matplotlib.cm.nipy_spectral,
                    edgecolor='none', linewidth=0.1)
    ax.set_title('surface')
    plt.show()
