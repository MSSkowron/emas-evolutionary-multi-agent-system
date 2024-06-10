import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LB = -5.12
UB = 5.12
funcName = "Rastrigin"


def rastrigin(x, a=10):
    dim = len(x)
    return a * dim + np.sum([(xi ** 2 - a * np.cos(2 * np.pi * xi)) for xi in x])


def generate_points(min_val, max_val, num_points):
    return np.linspace(min_val, max_val, num_points)


if __name__ == "__main__":
    dimension, amount_of_points = 2, 100
    X = generate_points(-5.12, 5.12, amount_of_points)
    Y = generate_points(-5.12, 5.12, amount_of_points)

    A = [[X[x] for _ in range(amount_of_points)]
         for x in range(amount_of_points)]
    B = [[Y[y] for y in range(amount_of_points)]
         for _ in range(amount_of_points)]
    C = [[rastrigin([X[x], Y[y]]) for y in range(amount_of_points)]
         for x in range(amount_of_points)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.array(A), np.array(B), np.array(C), rstride=1, cstride=1, cmap=matplotlib.cm.nipy_spectral,
                    edgecolor='none', linewidth=0.1)
    ax.set_title('surface')
    plt.show()
