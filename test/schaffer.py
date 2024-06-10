import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LB = -100
UB = 100
funcName = "Schaffer"


def schaffer(x):
    if len(x) == 0:
        x, y = 0, 0
    elif len(x) == 1:
        x, y = x[0], 0
    else:
        x, y = x[0], x[1]

    return 0.5 + (np.square(np.sin(np.square(x) - np.square(y))) - 0.5) / np.square(1 + 0.001 * (np.square(x) + np.square(y)))


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
    C = [[schaffer([X[x], Y[y]]) for y in range(amount_of_points)]
         for x in range(amount_of_points)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.array(A), np.array(B), np.array(C), rstride=1, cstride=1, cmap=matplotlib.cm.nipy_spectral,
                    edgecolor='none', linewidth=0.1)
    ax.set_title('surface')
    plt.show()
