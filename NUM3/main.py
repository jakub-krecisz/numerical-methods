import matplotlib.pyplot as plt
import numpy as np
import time


def get_solution_by_numpy_lib(size):
    aMatrix = np.diag([0.2] * (size - 1), -1)
    aMatrix += np.diag([1.2] * size)
    aMatrix += np.diag([0.1 / i for i in range(1, size)], 1)
    aMatrix += np.diag([0.4 / i ** 2 for i in range(1, size - 1)], 2)
    xVector = np.array([_ for _ in range(1, size + 1)])

    startTime = time.time()
    solution = np.linalg.solve(aMatrix, xVector)

    return solution, time.time() - startTime


def get_solution_numerically(size):
    aMatrix = [[0] + [0.2] * (size - 1), [1.2] * size, [0.1 / i for i in range(1, size)] + [0],
               [0.4 / i ** 2 for i in range(1, size - 1)] + [0] + [0]]

    xVector = [_ for _ in range(1, size + 1)]

    startTime = time.time()
    for i in range(1, size - 2):
        aMatrix[0][i] = aMatrix[0][i] / aMatrix[1][i - 1]
        aMatrix[1][i] = aMatrix[1][i] - aMatrix[0][i] * aMatrix[2][i - 1]
        aMatrix[2][i] = aMatrix[2][i] - aMatrix[0][i] * aMatrix[3][i - 1]

    aMatrix[0][size - 2] = aMatrix[0][size - 2] / aMatrix[1][size - 3]
    aMatrix[1][size - 2] = aMatrix[1][size - 2] - aMatrix[0][size - 2] * aMatrix[2][size - 3]
    aMatrix[2][size - 2] = aMatrix[2][size - 2] - aMatrix[0][size - 2] * aMatrix[3][size - 3]

    aMatrix[0][size - 1] = aMatrix[0][size - 1] / aMatrix[1][size - 2]
    aMatrix[1][size - 1] = aMatrix[1][size - 1] - aMatrix[0][size - 1] * aMatrix[2][size - 2]

    # Forward Substitution
    for i in range(1, size):
        xVector[i] = xVector[i] - aMatrix[0][i] * xVector[i - 1]

    # Backward Substitution
    xVector[size - 1] = xVector[size - 1] / aMatrix[1][size - 1]
    xVector[size - 2] = (xVector[size - 2] - aMatrix[2][size - 2] * xVector[size - 1]) / aMatrix[1][size - 2]
    for i in range(size - 3, -1, -1):
        xVector[i] = (xVector[i] - aMatrix[3][i] * xVector[i + 2] - aMatrix[2][i] * xVector[i + 1]) / aMatrix[1][i]

    # Determinant
    detA = 1
    for val in aMatrix[1]:
        detA *= val

    return xVector, detA, time.time() - startTime


def generate_graph():
    numpyResults = {}
    numericalResults = {}

    for size in range(100, 5000, 200):
        numpyResults[size] = get_solution_by_numpy_lib(size)[1] * 1000000
        numericalResults[size] = get_solution_numerically(size)[2] * 1000000

    plt.grid(True)
    plt.yscale('log')
    plt.title('Computing time')
    plt.xlabel('Matrix dimension (N)')
    plt.ylabel('Microseconds (μs)')
    plt.plot(numpyResults.keys(), numpyResults.values(), 'tab:green')
    plt.plot(numpyResults.keys(), numericalResults.values(), 'tab:red')
    plt.legend(['Solving time by numPy library', 'Solving time numerically'])
    plt.show()


if __name__ == '__main__':
    n = 100
    print(f'The solution of the equation for N = {n}:\n{get_solution_numerically(n)[0]}\n'
          f'The determinant of A is equal: det(A) = {get_solution_numerically(n)[1]}')
    generate_graph()
