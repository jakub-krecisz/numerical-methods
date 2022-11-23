import random

import numpy as np
import matplotlib.pyplot as plt

# Config
N = 100
MAX_ITERATION = 500
PRECISION = -12
X_VEC = random.sample(range(500), N)
B_VEC = list(range(1, N + 1))

def jacobin_alg(vectorX, y):
    for index in range(N):
        if index == 0:
            vectorX[index] = (B_VEC[index] - vectorX[index + 1] - 0.2 * vectorX[index + 2]) / 3
        elif index == 1:
            vectorX[index] = (B_VEC[index] - y[index - 1] - vectorX[index + 1] - 0.2 * vectorX[index + 2]) / 3
        elif index == N - 2:
            vectorX[index] = (B_VEC[index] - y[index - 1] - 0.2 * y[index - 2] - vectorX[index + 1]) / 3
        elif index == N - 1:
            vectorX[index] = (B_VEC[index] - y[index - 1] - 0.2 * y[index - 2]) / 3
        else:
            vectorX[index] = (B_VEC[index] - y[index - 1] - 0.2 * y[index - 2] - vectorX[index + 1] - 0.2 * vectorX[
                index + 2]) / 3
        return vectorX, y

def gauss_alg(vectorX):
    for index in range(N):
        if index == 0:
            vectorX[index] = (B_VEC[index] - vectorX[index + 1] - 0.2 * vectorX[index + 2]) / 3
        elif index == 1:
            vectorX[index] = (B_VEC[index] - vectorX[index - 1] - vectorX[index + 1] - 0.2 * vectorX[index + 2]) / 3
        elif index == N - 2:
            vectorX[index] = (B_VEC[index] - vectorX[index - 1] - 0.2 * vectorX[index - 2] - vectorX[index + 1]) / 3
        elif index == N - 1:
            vectorX[index] = (B_VEC[index] - vectorX[index - 1] - 0.2 * vectorX[index - 2]) / 3
        else:
            vectorX[index] = (B_VEC[index] - vectorX[index - 1] - 0.2 * vectorX[index - 2] - vectorX[
                index + 1] - 0.2 * vectorX[index + 2]) / 3
    return vectorX

# Jacobin Implementation
def get_solution_jacobin(vectorX, maxIteration, prec):
    prevNormVal = 0
    approximationsList = []
    while maxIteration:
        y = vectorX.copy()
        # Jacobin algorythm
        vectorX, y = jacobin_alg(vectorX, y)
        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, vectorX, y)))
        approximationsList.append(vectorX.copy())

        if abs(prevNormVal - actualNorm) < 10 ** prec:
            break
        prevNormVal = actualNorm
        maxIteration -= 1

    return approximationsList


# Gauss-Seidel implementation
def get_solution_gauss_seidel(vectorX, maxIteration, prec):
    prevNormVal = 0
    approximationsList = []
    while maxIteration:
        y = vectorX.copy()
        # Gauss-Seidel algorythm

        actualNorm = np.sqrt(sum(map(lambda a, b: (a - b) ** 2, vectorX, y)))
        approximationsList.append(vectorX.copy())

        if abs(prevNormVal - actualNorm) < 10 ** prec:
            break

        prevNormVal = actualNorm
        maxIteration = maxIteration - 1

    return approximationsList


def generate_graph():
    approximationsJacobin = get_solution_jacobin(X_VEC.copy(), MAX_ITERATION, PRECISION)
    approximationsGaussSeidel = get_solution_gauss_seidel(X_VEC.copy(), MAX_ITERATION, PRECISION)

    diffs1, diffs2 = [], []
    for i in range(len(approximationsJacobin) - 1):
        diffs1.append(np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximationsJacobin[i], approximationsJacobin[-1]))))
    for i in range(len(approximationsGaussSeidel) - 1):
        diffs2.append(
            np.sqrt(sum(map(lambda a, b: (a - b) ** 2, approximationsGaussSeidel[i], approximationsGaussSeidel[-1]))))

    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel("$|x(n) - x.last|$")
    plt.yscale('log')
    plt.plot(list(range(len(diffs1))), diffs1)
    plt.plot(list(range(len(diffs2))), diffs2)
    plt.legend(['Jacobin', 'Gauss-Seidel'])
    plt.title('Comparison of two iterative methods\nbetween their differences of Approximation')
    plt.show()


print(f'Jacobin solution: \n{get_solution_jacobin(X_VEC.copy(), MAX_ITERATION, PRECISION)[-1]}')
print(f'Gauss-Seidel solution: \n{get_solution_gauss_seidel(X_VEC.copy(), MAX_ITERATION, PRECISION)[-1]}')
generate_graph()
