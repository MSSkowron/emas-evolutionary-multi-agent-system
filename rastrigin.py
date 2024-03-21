import numpy as np
import matplotlib.pyplot as plt
import math
import random
import matplotlib

def rastrigin(X, dim = 2, A = 10):
    if len(X) != dim: return 0
    return A * dim + np.sum([ (x**2 - A*math.cos(2*math.pi*x)) for x in X])
        

if __name__ == "__main__":
    
    dimension = 2
    amount_of_points = 100
    X = np.linspace(-5.12, 5.12, amount_of_points)
    Y = np.linspace(-5.12, 5.12, amount_of_points)
    A = []
    B = []
    C = []

    for x in range(amount_of_points):
        A.append([])
        B.append([])
        C.append([])
        for y in range(amount_of_points):
            A[x].append(X[x])
            B[x].append(Y[y])
            result = rastrigin([X[x],Y[y]])
            C[x].append(result)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.array(A), np.array(B), np.array(C), rstride=1, cstride=1,cmap=matplotlib.cm.nipy_spectral, edgecolor='none', linewidth=0.1)
    ax.set_title('surface')
    plt.show()